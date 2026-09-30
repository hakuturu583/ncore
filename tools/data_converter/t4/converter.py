# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T4 dataset to NCore V4 converter.

T4 ``base_link`` and ``map`` frames map to NCore ``rig`` and ``world``.
``calibrated_sensor`` stores ``T_sensor_baselink`` and ``ego_pose`` stores
``T_baselink_map``; both match NCore's source->target convention with no
inversion needed.

Optionally imports gaussian_factory per-frame artifacts (instance / sky masks,
accumulated LiDAR depth, MapAnything depth, refined camera poses) so that the
raw T4 tree can be replaced by the NCore sequence.
"""

from __future__ import annotations

import io
import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import click
import numpy as np
import tqdm

from PIL import Image as PILImage
from scipy.spatial.transform import Rotation
from upath import UPath

from ncore.impl.common.transformations import HalfClosedInterval
from ncore.impl.data.types import (
    BBox3,
    CuboidTrackObservation,
    JsonLike,
    LabelSource,
    OpenCVPinholeCameraModelParameters,
    PointCloud,
    ShutterType,
)
from ncore.impl.data.v4.components import (
    CameraLabelsComponent,
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    MasksComponent,
    PointCloudsComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.impl.data.v4.types import ComponentGroupAssignments
from ncore.impl.data_converter.base import FileBasedDataConverter, FileBasedDataConverterConfig
from tools.data_converter.cli import cli
from tools.data_converter.t4.gaussian_factory import (
    GF_LAYERS,
    CameraFrameRef,
    DepthEncoding,
    GfLayerImporter,
    load_camera_poses_json,
    resolve_layers,
)
from tools.data_converter.t4.utils import (
    index_by_token,
    keyframe_index_by_sample_token,
    load_annotation_tables,
    load_lidar_points,
    t4_distortion_to_opencv,
    t4_pose_to_se3,
)


T4_LIDAR_INTENSITY_MAX: float = 255.0

LidarFormat = Literal["ray-bundle", "point-cloud"]


@dataclass(kw_only=True, slots=True)
class T4Converter4Config(FileBasedDataConverterConfig):
    """Configuration for T4 to NCore V4 conversion."""

    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True
    label_source: Literal["autolabel", "gt-annotation", "external"] = "autolabel"
    # Optional directory of rebuilt LIDAR_CONCAT frames (NNNNN.bin, float32 x7:
    # x,y,z,intensity,ring,return_type,time_sec) produced by decoding the raw
    # pandar_packets. When set, the lidar component uses these real ring +
    # per-point timestamps instead of the T4 .pcd.bin (which lacks both).
    rebuilt_lidar_dir: Optional[str] = None

    ## Storage reduction
    keyframes_only: bool = False  # only convert sample_data with is_key_frame=True
    lidar_format: LidarFormat = "ray-bundle"
    jpeg_quality: Optional[int] = None  # re-encode camera JPEGs at this quality (lossy); None keeps source bytes
    include_lanelet2_map: bool = True  # embed map/lanelet2_map.osm (if present) as poses generic data

    ## gaussian_factory artifacts
    gf_root: Optional[str] = None  # gaussian_factory output root of this scene (holds sam_masks/, sky_masks/, ...)
    gf_layers: Tuple[str, ...] = ()  # subset of layers to import (all found if empty)
    gf_layer_dirs: Tuple[str, ...] = ()  # NAME=PATH overrides of layer directories
    gf_depth_encoding: DepthEncoding = "uint16"
    gf_png_reoptimize: bool = True
    gf_camera_poses_json: Optional[str] = None  # {sample_data_token: 4x4 camera->world} (cuVSLAM / refined poses)


class T4Converter4(FileBasedDataConverter):
    """T4 dataset to NCore V4 converter."""

    _LABEL_SOURCE_MAP = {
        "autolabel": LabelSource.AUTOLABEL,
        "gt-annotation": LabelSource.GT_ANNOTATION,
        "external": LabelSource.EXTERNAL,
    }

    def __init__(self, config: T4Converter4Config) -> None:
        super().__init__(config)
        self.component_group_profile = config.component_group_profile
        self.store_type = config.store_type
        self.store_sequence_meta = config.store_sequence_meta
        self.label_source = self._LABEL_SOURCE_MAP[config.label_source]
        self.rebuilt_lidar_dir = Path(config.rebuilt_lidar_dir) if config.rebuilt_lidar_dir else None
        self.keyframes_only = config.keyframes_only
        self.lidar_format: LidarFormat = config.lidar_format
        self.jpeg_quality = config.jpeg_quality
        self.include_lanelet2_map = config.include_lanelet2_map
        self.gf_root = Path(config.gf_root) if config.gf_root else None
        self.gf_layers = list(config.gf_layers)
        self.gf_layer_dirs: Dict[str, Path] = {}
        for spec in config.gf_layer_dirs:
            name, sep, path = spec.partition("=")
            if not sep:
                raise ValueError(f"--gf-layer-dir expects NAME=PATH, got {spec!r}")
            self.gf_layer_dirs[name] = Path(path)
        self.gf_depth_encoding: DepthEncoding = config.gf_depth_encoding
        self.gf_png_reoptimize = config.gf_png_reoptimize
        self.gf_camera_poses_json = Path(config.gf_camera_poses_json) if config.gf_camera_poses_json else None
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_sequence_ids(config: T4Converter4Config) -> list[str]:
        """A T4 sequence is a directory holding ``annotation/`` and ``data/``."""
        root = Path(config.root_dir)

        def _is_t4_sequence(p: Path) -> bool:
            return (p / "annotation").is_dir() and (p / "data").is_dir()

        if _is_t4_sequence(root):
            return [str(root)]

        return [str(p) for p in sorted(root.iterdir()) if p.is_dir() and _is_t4_sequence(p)]

    @staticmethod
    def from_config(config: T4Converter4Config) -> T4Converter4:
        return T4Converter4(config)

    def convert_sequence(self, sequence_id: str) -> None:
        sequence_path = Path(sequence_id)
        sequence_name = sequence_path.name
        self.logger.info(f"Converting T4 sequence: {sequence_name}")

        # Lidar scan period drives the per-frame time window. Read from
        # status.json when present; fall back to 100 ms otherwise.
        status_path = sequence_path / "status.json"
        lidar_scan_period_us = 100_000
        if status_path.exists():
            with status_path.open("r") as f:
                status_root = json.load(f)
            status_inner = next(iter(status_root.values())) if status_root else {}
            period_sec = status_inner.get("_lidar_scan_period_sec")
            if period_sec is not None:
                lidar_scan_period_us = int(round(float(period_sec) * 1_000_000))

        tables = load_annotation_tables(sequence_path / "annotation")
        scene = tables["scene"][0]
        sensors_by_token = index_by_token(tables["sensor"])
        calibrated_by_token = index_by_token(tables["calibrated_sensor"])
        keyframe_index = keyframe_index_by_sample_token(tables["sample"], scene)

        def _modality(sd: dict) -> str:
            return sensors_by_token[calibrated_by_token[sd["calibrated_sensor_token"]]["sensor_token"]]["modality"]

        sample_data_by_channel: Dict[str, list[dict]] = {}
        for sd in tables["sample_data"]:
            calib = calibrated_by_token[sd["calibrated_sensor_token"]]
            channel = sensors_by_token[calib["sensor_token"]]["channel"]
            sample_data_by_channel.setdefault(channel, []).append(sd)
        # ordinal among all frames of the channel (lidar_rebuild names frames by it)
        channel_ordinal: Dict[str, int] = {}
        for channel in sample_data_by_channel:
            sample_data_by_channel[channel].sort(key=lambda sd: sd["timestamp"])
            channel_ordinal.update({sd["token"]: i for i, sd in enumerate(sample_data_by_channel[channel])})
            if self.keyframes_only:
                sample_data_by_channel[channel] = [
                    sd for sd in sample_data_by_channel[channel] if sd.get("is_key_frame", False)
                ]
        sample_data_by_channel = {ch: sds for ch, sds in sample_data_by_channel.items() if sds}

        camera_channels = sorted(ch for ch, sds in sample_data_by_channel.items() if _modality(sds[0]) == "camera")
        lidar_channels = sorted(ch for ch, sds in sample_data_by_channel.items() if _modality(sds[0]) == "lidar")

        camera_ids = self.get_active_camera_ids(camera_channels)
        lidar_ids = self.get_active_lidar_ids(lidar_channels)

        # Sequence interval must contain every per-frame window and ego pose we store.
        # ego_pose may start before / end after the sample_data range (e.g. the
        # tier4_perception_dataset sample).
        stored_sds = [sd for ch in camera_ids + lidar_ids for sd in sample_data_by_channel[ch]]
        sd_ts = [sd["timestamp"] for sd in stored_sds] or [sd["timestamp"] for sd in tables["sample_data"]]
        lidar_ts = [sd["timestamp"] for ch in lidar_ids for sd in sample_data_by_channel[ch]]
        ego_ts = [r["timestamp"] for r in tables["ego_pose"]]
        seq_start_us = min(min(sd_ts), min(ego_ts))
        seq_end_us_inclusive = max(max(sd_ts), max(ego_ts))
        if lidar_ts:
            seq_end_us_inclusive = max(seq_end_us_inclusive, max(lidar_ts) + lidar_scan_period_us - 1)
        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(seq_start_us, seq_end_us_inclusive)

        ego_records_sorted = sorted(tables["ego_pose"], key=lambda r: r["timestamp"])
        ego_records_dedup: list[dict] = []
        last_ts = None
        for r in ego_records_sorted:
            if r["timestamp"] == last_ts:
                continue
            ego_records_dedup.append(r)
            last_ts = r["timestamp"]

        ego_timestamps_us = np.array([r["timestamp"] for r in ego_records_dedup], dtype=np.uint64)
        T_rig_world = np.stack(
            [t4_pose_to_se3(r["translation"], r["rotation"]) for r in ego_records_dedup],
            axis=0,
        )

        # Replicate boundary poses so the trajectory covers the full interval.
        if ego_timestamps_us[0] > seq_start_us:
            ego_timestamps_us = np.concatenate([np.array([seq_start_us], dtype=np.uint64), ego_timestamps_us])
            T_rig_world = np.concatenate([T_rig_world[:1], T_rig_world], axis=0)
        if ego_timestamps_us[-1] < seq_end_us_inclusive:
            ego_timestamps_us = np.concatenate([ego_timestamps_us, np.array([seq_end_us_inclusive], dtype=np.uint64)])
            T_rig_world = np.concatenate([T_rig_world, T_rig_world[-1:]], axis=0)

        gf_importers = resolve_layers(
            self.gf_root, self.gf_layers, self.gf_layer_dirs, self.gf_depth_encoding, self.gf_png_reoptimize
        )
        # (importer, camera_id) -> camera label instance name
        gf_label_instances: Dict[Tuple[str, str], str] = {}
        for importer in gf_importers:
            for camera_id in camera_ids:
                if importer.has_camera(camera_id):
                    gf_label_instances[(importer.layer.name, camera_id)] = importer.descriptor(
                        camera_id
                    ).default_instance_name

        point_cloud_lidar_ids = lidar_ids if self.lidar_format == "point-cloud" else []
        component_groups = ComponentGroupAssignments.create(
            camera_ids=camera_ids,
            lidar_ids=[] if self.lidar_format == "point-cloud" else lidar_ids,
            radar_ids=[],
            point_clouds_ids=point_cloud_lidar_ids,
            camera_labels_ids=list(gf_label_instances.values()),
            profile=self.component_group_profile,
            # one store per gaussian_factory layer, so layers can be dropped / shipped independently
            camera_labels_component_groups={
                instance: f"gf_{layer_name}" for (layer_name, _), instance in gf_label_instances.items()
            },
        )

        sequence_meta: Dict[str, JsonLike] = {
            "source_format": "t4",
            "t4_scene_token": scene["token"],
            "t4_scene_name": scene.get("name", ""),
            "t4_log_token": scene["log_token"],
            "t4_keyframes_only": self.keyframes_only,
            "t4_lidar_format": self.lidar_format,
            "t4_jpeg_quality": self.jpeg_quality,
            "gf_layers": list(sorted({layer for layer, _ in gf_label_instances})),
        }
        store_writer = SequenceComponentGroupsWriter(
            output_dir_path=UPath(self.output_dir) / sequence_name,
            store_base_name=sequence_name,
            sequence_id=sequence_name,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data=sequence_meta,
        )

        poses_writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.poses_component_group,
            generic_meta_data={"calibration_type": "t4:calibrated_sensor", "egomotion_type": "t4:ego_pose"},
        )
        intrinsics_writer = store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.intrinsics_component_group,
        )
        masks_writer = store_writer.register_component_writer(
            MasksComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.masks_component_group,
        )

        poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=T_rig_world.astype(np.float32),
            timestamps_us=ego_timestamps_us,
        )

        # T4 ``map`` is itself a metric global frame (e.g. UTM), so
        # ``world -> world_global`` is identity. Downstream tools expect this
        # transform to be present and to be stored in float64.
        poses_writer.store_static_pose(
            source_frame_id="world",
            target_frame_id="world_global",
            pose=np.eye(4, dtype=np.float64),
        )

        lanelet2_path = sequence_path / "map" / "lanelet2_map.osm"
        if self.include_lanelet2_map and lanelet2_path.is_file():
            poses_writer.set_generic_data(
                {"t4_lanelet2_map_osm": np.frombuffer(lanelet2_path.read_bytes(), dtype=np.uint8)},
                meta_data={"t4_lanelet2_map_osm": "utf-8 bytes of map/lanelet2_map.osm (world frame)"},
            )

        for lidar_id in lidar_ids:
            self._convert_lidar(
                lidar_id=lidar_id,
                sequence_path=sequence_path,
                sample_data=sample_data_by_channel[lidar_id],
                calibrated_by_token=calibrated_by_token,
                keyframe_index=keyframe_index,
                channel_ordinal=channel_ordinal,
                store_writer=store_writer,
                poses_writer=poses_writer,
                component_groups=component_groups,
                scan_period_us=lidar_scan_period_us,
            )

        camera_frames: Dict[str, List[CameraFrameRef]] = {}
        for camera_id in camera_ids:
            camera_frames[camera_id] = self._convert_camera(
                camera_id=camera_id,
                sequence_path=sequence_path,
                sample_data=sample_data_by_channel[camera_id],
                calibrated_by_token=calibrated_by_token,
                keyframe_index=keyframe_index,
                store_writer=store_writer,
                poses_writer=poses_writer,
                intrinsics_writer=intrinsics_writer,
                masks_writer=masks_writer,
                component_groups=component_groups,
            )

        for importer in gf_importers:
            self._convert_gf_layer(importer, camera_frames, gf_label_instances, store_writer, component_groups)

        if self.gf_camera_poses_json is not None:
            self._convert_gf_camera_poses(camera_frames, store_writer, component_groups)

        if tables["sample_annotation"]:
            self._convert_cuboids(
                tables=tables,
                store_writer=store_writer,
                component_groups=component_groups,
            )

        ncore_paths = store_writer.finalize()

        if self.store_sequence_meta:
            reader = SequenceComponentGroupsReader(ncore_paths)
            meta_path = UPath(self.output_dir) / sequence_name / f"{reader.sequence_id}.json"
            with meta_path.open("w") as f:
                json.dump(reader.get_sequence_meta().to_dict(), f, indent=2)

    @staticmethod
    def _frame_meta(sd: dict, keyframe_index: Dict[str, int]) -> Dict[str, JsonLike]:
        """Per-frame T4 provenance, so artifacts keyed by token / index stay resolvable."""
        return {
            "t4_sample_data_token": sd["token"],
            "t4_sample_token": sd.get("sample_token", ""),
            "t4_filename": sd["filename"],
            "t4_is_key_frame": bool(sd.get("is_key_frame", False)),
            "t4_keyframe_index": keyframe_index.get(sd.get("sample_token", "")) if sd.get("is_key_frame") else None,
        }

    def _convert_lidar(
        self,
        lidar_id: str,
        sequence_path: Path,
        sample_data: list[dict],
        calibrated_by_token: dict,
        keyframe_index: Dict[str, int],
        channel_ordinal: Dict[str, int],
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        component_groups: ComponentGroupAssignments,
        scan_period_us: int,
    ) -> None:
        calib = calibrated_by_token[sample_data[0]["calibrated_sensor_token"]]
        T_sensor_rig = t4_pose_to_se3(calib["translation"], calib["rotation"])
        poses_writer.store_static_pose(
            source_frame_id=lidar_id,
            target_frame_id="rig",
            pose=T_sensor_rig.astype(np.float32),
        )

        use_rebuilt = self.rebuilt_lidar_dir is not None and lidar_id == "LIDAR_CONCAT"
        if use_rebuilt:
            self.logger.info(f"Using rebuilt lidar (real ring + per-point time) from {self.rebuilt_lidar_dir}")

        if self.lidar_format == "point-cloud":
            attribute_schemas = {
                "intensity": PointCloudsComponent.AttributeSchema(
                    transform_type=PointCloud.AttributeTransformType.INVARIANT, dtype=np.dtype("uint8")
                ),
                "ring": PointCloudsComponent.AttributeSchema(
                    transform_type=PointCloud.AttributeTransformType.INVARIANT, dtype=np.dtype("uint16")
                ),
            }
            if use_rebuilt:
                attribute_schemas["timestamp_us"] = PointCloudsComponent.AttributeSchema(
                    transform_type=PointCloud.AttributeTransformType.INVARIANT, dtype=np.dtype("uint64")
                )
            pc_writer = store_writer.register_component_writer(
                PointCloudsComponent.Writer,
                component_instance_name=lidar_id,
                group_name=component_groups.point_clouds_component_groups.get(lidar_id),
                generic_meta_data={"source": "t4:lidar", "intensity_range": [0, 255]},
                coordinate_unit=PointCloud.CoordinateUnit.METERS,
                attribute_schemas=attribute_schemas,
            )
        else:
            lidar_writer = store_writer.register_component_writer(
                LidarSensorComponent.Writer,
                component_instance_name=lidar_id,
                group_name=component_groups.lidar_component_groups.get(lidar_id),
                generic_meta_data={},
            )

        for sd in tqdm.tqdm(sample_data, desc=f"lidar {lidar_id}"):
            ts_us = np.uint64(sd["timestamp"])
            frame_meta = self._frame_meta(sd, keyframe_index)

            point_time_sec = None
            if use_rebuilt:
                rebuilt = self.rebuilt_lidar_dir / f"{channel_ordinal[sd['token']]:05d}.bin"
                if not rebuilt.exists():
                    raise FileNotFoundError(f"Rebuilt lidar frame missing: {rebuilt}")
                # x, y, z, intensity(0-255), ring, return_type, time_sec_offset
                rec = np.fromfile(rebuilt, dtype=np.float32).reshape(-1, 7)
                xyz = rec[:, :3]
                intensity_raw = rec[:, 3]
                ring = rec[:, 4]
                point_time_sec = rec[:, 6]
            else:
                points = load_lidar_points(sequence_path / sd["filename"])
                xyz = points[:, :3]
                intensity_raw = points[:, 3]
                # 5th T4 channel is the ring / laser index
                ring = points[:, 4] if points.shape[1] > 4 else None

            distance_m = np.linalg.norm(xyz, axis=1).astype(np.float32)
            valid = distance_m > 0
            # NCore requires unit-norm directions; drop zero-distance rays.
            if not valid.all():
                xyz, distance_m, intensity_raw = xyz[valid], distance_m[valid], intensity_raw[valid]
                ring = ring[valid] if ring is not None else None
                point_time_sec = point_time_sec[valid] if point_time_sec is not None else None
            n_rays = xyz.shape[0]

            if point_time_sec is not None:
                # Real per-point timestamps: frame header (= earliest point) + per-point offset.
                point_timestamps_us = (ts_us + (point_time_sec * 1e6).astype(np.uint64)).astype(np.uint64)
                frame_end_us = point_timestamps_us.max() if n_rays else ts_us + np.uint64(scan_period_us - 1)
                point_timestamps_us = np.clip(point_timestamps_us, ts_us, frame_end_us)
            else:
                point_timestamps_us = np.full(n_rays, ts_us, dtype=np.uint64)
                frame_end_us = ts_us + np.uint64(scan_period_us - 1)

            ring_u16 = (
                np.clip(np.round(ring), 0, np.iinfo(np.uint16).max).astype(np.uint16)
                if ring is not None
                else np.zeros(n_rays, dtype=np.uint16)
            )

            if self.lidar_format == "point-cloud":
                attributes = {
                    "intensity": np.clip(np.round(intensity_raw), 0, 255).astype(np.uint8),
                    "ring": ring_u16,
                }
                if use_rebuilt:
                    attributes["timestamp_us"] = point_timestamps_us
                pc_writer.store_pc(
                    xyz=np.ascontiguousarray(xyz, dtype=np.float32),
                    reference_frame_id=lidar_id,
                    reference_frame_timestamp_us=int(ts_us),
                    attributes=attributes,
                    generic_meta_data=frame_meta,
                )
                continue

            direction = (xyz / distance_m[:, None]).astype(np.float32)
            intensity = np.clip(intensity_raw / T4_LIDAR_INTENSITY_MAX, 0.0, 1.0).astype(np.float32)
            # The concatenated cloud fuses several physical lidars and has no single
            # structured row/column grid, so it is stored unstructured
            # (model_element=None); the T4 ring index is kept as generic data.
            lidar_writer.store_frame(
                direction=direction,
                timestamp_us=point_timestamps_us,
                model_element=None,
                distance_m=distance_m.reshape(1, -1),
                intensity=intensity.reshape(1, -1),
                frame_timestamps_us=np.array([ts_us, frame_end_us], dtype=np.uint64),
                generic_data={"ring": ring_u16} if ring is not None else {},
                generic_meta_data=frame_meta,
            )

    def _encode_image(self, image_binary: bytes) -> bytes:
        if self.jpeg_quality is None:
            return image_binary
        with PILImage.open(io.BytesIO(image_binary)) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="jpeg", quality=self.jpeg_quality, optimize=True)
        reencoded = buf.getvalue()
        return reencoded if len(reencoded) < len(image_binary) else image_binary

    def _convert_camera(
        self,
        camera_id: str,
        sequence_path: Path,
        sample_data: list[dict],
        calibrated_by_token: dict,
        keyframe_index: Dict[str, int],
        store_writer: SequenceComponentGroupsWriter,
        poses_writer: PosesComponent.Writer,
        intrinsics_writer: IntrinsicsComponent.Writer,
        masks_writer: MasksComponent.Writer,
        component_groups: ComponentGroupAssignments,
    ) -> List[CameraFrameRef]:
        calib = calibrated_by_token[sample_data[0]["calibrated_sensor_token"]]
        T_sensor_rig = t4_pose_to_se3(calib["translation"], calib["rotation"])
        poses_writer.store_static_pose(
            source_frame_id=camera_id,
            target_frame_id="rig",
            pose=T_sensor_rig.astype(np.float32),
        )

        K = np.array(calib["camera_intrinsic"], dtype=np.float32)
        fu, fv = float(K[0, 0]), float(K[1, 1])
        cu, cv = float(K[0, 2]), float(K[1, 2])
        radial, tangential, thin_prism = t4_distortion_to_opencv(calib.get("camera_distortion") or [])

        width = int(sample_data[0].get("width") or 0)
        height = int(sample_data[0].get("height") or 0)
        if width == 0 or height == 0:
            raise ValueError(f"Camera {camera_id}: sample_data is missing width/height")

        camera_writer = store_writer.register_component_writer(
            CameraSensorComponent.Writer,
            component_instance_name=camera_id,
            group_name=component_groups.camera_component_groups.get(camera_id),
            generic_meta_data={},
        )

        frames: List[CameraFrameRef] = []
        for sd in tqdm.tqdm(sample_data, desc=f"camera {camera_id}"):
            img_path = sequence_path / sd["filename"]
            with img_path.open("rb") as f:
                image_binary = self._encode_image(f.read())
            ts_us = np.uint64(sd["timestamp"])
            frame_meta = self._frame_meta(sd, keyframe_index)
            camera_writer.store_frame(
                image_binary_data=image_binary,
                image_format="jpeg",
                frame_timestamps_us=np.array([ts_us, ts_us], dtype=np.uint64),
                generic_data={},
                generic_meta_data=frame_meta,
            )
            frames.append(
                CameraFrameRef(
                    camera_id=camera_id,
                    timestamp_us=int(ts_us),
                    sample_data_token=sd["token"],
                    image_stem=Path(sd["filename"]).stem,
                    keyframe_index=keyframe_index.get(sd.get("sample_token", "")) if sd.get("is_key_frame") else None,
                )
            )

        intrinsics_writer.store_camera_intrinsics(
            camera_id=camera_id,
            camera_model_parameters=OpenCVPinholeCameraModelParameters(
                resolution=np.array([width, height], dtype=np.uint64),
                shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
                external_distortion_parameters=None,
                principal_point=np.array([cu, cv], dtype=np.float32),
                focal_length=np.array([fu, fv], dtype=np.float32),
                radial_coeffs=radial,
                tangential_coeffs=tangential,
                thin_prism_coeffs=thin_prism,
            ),
        )
        masks_writer.store_camera_masks(camera_id=camera_id, mask_images={})
        return frames

    def _convert_gf_layer(
        self,
        importer: GfLayerImporter,
        camera_frames: Dict[str, List[CameraFrameRef]],
        gf_label_instances: Dict[Tuple[str, str], str],
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
    ) -> None:
        layer = importer.layer
        component_meta = importer.component_meta()
        for camera_id, frames in camera_frames.items():
            instance = gf_label_instances.get((layer.name, camera_id))
            if instance is None:
                continue
            writer = store_writer.register_component_writer(
                CameraLabelsComponent.Writer,
                component_instance_name=instance,
                group_name=component_groups.camera_labels_component_groups.get(instance),
                generic_meta_data=component_meta,
                descriptor=importer.descriptor(camera_id),
            )
            n_stored = n_missing = 0
            for frame in tqdm.tqdm(frames, desc=f"gf {layer.name} {camera_id}"):
                path = importer.path_for(frame)
                if path is None or not path.is_file():
                    n_missing += 1
                    continue
                writer.store_label(
                    importer.load(path),
                    timestamp_us=frame.timestamp_us,
                    generic_meta_data={"t4_sample_data_token": frame.sample_data_token},
                )
                n_stored += 1
            self.logger.info(
                f"gaussian_factory {layer.name}@{camera_id}: stored {n_stored}, no artifact for {n_missing} frames"
            )
        if importer.n_depth_clipped:
            self.logger.warning(
                f"gaussian_factory {layer.name}: {importer.n_depth_clipped} depth px beyond uint16 range set invalid"
            )

    def _convert_gf_camera_poses(
        self,
        camera_frames: Dict[str, List[CameraFrameRef]],
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
    ) -> None:
        """Store gaussian_factory camera->world poses as a separate ``gaussian_factory`` poses instance."""
        assert self.gf_camera_poses_json is not None
        poses_by_token, meta = load_camera_poses_json(self.gf_camera_poses_json)
        poses_meta: Dict[str, JsonLike] = {
            "producer": "gaussian_factory",
            "source_file": self.gf_camera_poses_json.name,
            "description": "per-camera-frame camera->world (T4 map) poses, OpenCV camera axes",
            "gf_meta": meta,
        }
        writer = store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="gaussian_factory",
            group_name=component_groups.poses_component_group,
            generic_meta_data=poses_meta,
        )
        for camera_id, frames in camera_frames.items():
            matched = [
                (f.timestamp_us, poses_by_token[f.sample_data_token])
                for f in frames
                if f.sample_data_token in poses_by_token
            ]
            if len(matched) < 2:
                self.logger.warning(f"gaussian_factory poses {camera_id}: {len(matched)} matched frame(s), skipping")
                continue
            writer.store_dynamic_pose(
                source_frame_id=camera_id,
                target_frame_id="world",
                poses=np.stack([m[1] for m in matched]).astype(np.float64),
                timestamps_us=np.array([m[0] for m in matched], dtype=np.uint64),
                # gaussian_factory only estimates poses at (key)frame times
                require_sequence_time_coverage=False,
            )
            self.logger.info(f"gaussian_factory poses {camera_id}: {len(matched)}/{len(frames)} frames")

    def _convert_cuboids(
        self,
        tables: dict,
        store_writer: SequenceComponentGroupsWriter,
        component_groups: ComponentGroupAssignments,
    ) -> None:
        """Convert ``sample_annotation.json`` to CuboidTrackObservations in the world frame.

        T4 follows nuScenes: translation+rotation are in the global frame and
        ``size`` is ``[width, length, height]`` in box-local axes (length along
        local x, width along local y, height along local z). NCore's BBox3.dim
        is ``[dim_x, dim_y, dim_z]`` so we reorder to ``[length, width, height]``.

        Fields without a CuboidTrackObservation slot (``num_lidar_pts``,
        ``num_radar_pts``, visibility) are stored as component generic data aligned
        with the observation order.
        """
        sample_ts_by_token = {r["token"]: r["timestamp"] for r in tables["sample"]}
        category_name_by_token = {r["token"]: r["name"] for r in tables["category"]}
        category_token_by_instance = {r["token"]: r["category_token"] for r in tables["instance"]}
        visibility_level_by_token = {r["token"]: r.get("level", "") for r in tables["visibility"]}

        observations: list[CuboidTrackObservation] = []
        num_lidar_pts: list[int] = []
        num_radar_pts: list[int] = []
        visibility_levels: list[str] = []
        for ann in tables["sample_annotation"]:
            sample_ts = sample_ts_by_token[ann["sample_token"]]
            cat_token = category_token_by_instance.get(ann["instance_token"])
            class_id = category_name_by_token.get(cat_token, "unknown")

            tx, ty, tz = ann["translation"]
            width, length, height = ann["size"]
            qw, qx, qy, qz = ann["rotation"]
            rx, ry, rz = Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)

            observations.append(
                CuboidTrackObservation(
                    track_id=ann["instance_token"],
                    class_id=class_id,
                    timestamp_us=int(sample_ts),
                    reference_frame_id="world",
                    reference_frame_timestamp_us=int(sample_ts),
                    bbox3=BBox3(
                        centroid=(float(tx), float(ty), float(tz)),
                        dim=(float(length), float(width), float(height)),
                        rot=(float(rx), float(ry), float(rz)),
                    ),
                    source=self.label_source,
                )
            )
            num_lidar_pts.append(int(ann.get("num_lidar_pts", -1)))
            num_radar_pts.append(int(ann.get("num_radar_pts", -1)))
            visibility_levels.append(visibility_level_by_token.get(ann.get("visibility_token", ""), ""))

        level_names = sorted(set(visibility_levels))
        cuboids_writer = store_writer.register_component_writer(
            CuboidsComponent.Writer,
            component_instance_name="default",
            group_name=component_groups.cuboid_track_observations_component_group,
        )
        cuboids_writer.set_generic_data(
            {
                "t4_num_lidar_pts": np.array(num_lidar_pts, dtype=np.int32),
                "t4_num_radar_pts": np.array(num_radar_pts, dtype=np.int32),
                "t4_visibility_index": np.array([level_names.index(v) for v in visibility_levels], dtype=np.uint8),
            },
            meta_data={"t4_visibility_levels": list[JsonLike](level_names)},
        )
        cuboids_writer.store_observations(observations)


@cli.command()
@click.option(
    "--store-type",
    type=click.Choice(["itar", "directory"], case_sensitive=False),
    default="itar",
    show_default=True,
)
@click.option(
    "component_group_profile",
    "--profile",
    type=click.Choice(["default", "separate-sensors", "separate-all"], case_sensitive=False),
    default="separate-sensors",
    show_default=True,
)
@click.option("store_sequence_meta", "--sequence-meta/--no-sequence-meta", default=True)
@click.option(
    "--label-source",
    type=click.Choice(["autolabel", "gt-annotation", "external"], case_sensitive=False),
    default="autolabel",
    show_default=True,
    help="Provenance to record for cuboids in sample_annotation.json.",
)
@click.option(
    "--rebuilt-lidar-dir",
    type=str,
    default=None,
    help="Directory of rebuilt LIDAR_CONCAT frames (NNNNN.bin, float32 x7) with real "
    "ring + per-point time, decoded from raw pandar_packets. Overrides the T4 .pcd.bin.",
)
@click.option(
    "--keyframes-only/--all-frames",
    default=False,
    show_default=True,
    help="Only convert keyframe sample_data (is_key_frame=True). gaussian_factory uses keyframes only.",
)
@click.option(
    "--lidar-format",
    type=click.Choice(["ray-bundle", "point-cloud"]),
    default="ray-bundle",
    show_default=True,
    help="ray-bundle: NCore LidarSensorComponent (float32 direction/distance/intensity). "
    "point-cloud: PointCloudsComponent with float32 xyz + uint8 intensity + uint16 ring (~30%% smaller).",
)
@click.option(
    "--jpeg-quality",
    type=click.IntRange(1, 100),
    default=None,
    help="Re-encode camera JPEGs at this quality (lossy). Default keeps the source bytes.",
)
@click.option("--lanelet2-map/--no-lanelet2-map", "include_lanelet2_map", default=True, show_default=True)
@click.option(
    "--gf-root",
    type=str,
    default=None,
    help="gaussian_factory output root of this scene (containing sam_masks/, sky_masks/, lidar_depth_accum/, ...). "
    "Every layer found there is imported as NCore camera labels.",
)
@click.option(
    "--gf-layer",
    "gf_layers",
    multiple=True,
    type=click.Choice(sorted(GF_LAYERS)),
    help="Restrict to these gaussian_factory layers (multiple; default: all found).",
)
@click.option(
    "--gf-layer-dir",
    "gf_layer_dirs",
    multiple=True,
    type=str,
    help="Override a layer directory as NAME=PATH (multiple), e.g. mapanything_depth=/run/ma/view_depth.",
)
@click.option(
    "--gf-depth-encoding",
    type=click.Choice(["uint16", "float32"]),
    default="uint16",
    show_default=True,
    help="uint16 quantizes depth to 1/256 m (range 256 m); float32 is lossless.",
)
@click.option("--gf-png-reoptimize/--no-gf-png-reoptimize", default=True, show_default=True)
@click.option(
    "--gf-camera-poses-json",
    type=str,
    default=None,
    help="gaussian_factory {sample_data_token: camera->world} poses (trajectory_correction/poses.json or "
    "refined_poses_step*.json), stored as a 'gaussian_factory' poses component.",
)
@click.pass_context
def t4_v4(ctx, *_, **kwargs):
    """T4 dataset conversion (V4 format)"""
    kwargs["gf_layers"] = tuple(kwargs["gf_layers"])
    kwargs["gf_layer_dirs"] = tuple(kwargs["gf_layer_dirs"])
    config = T4Converter4Config(**{**vars(ctx.obj), **kwargs})
    T4Converter4.convert(config)
