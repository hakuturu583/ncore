# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T4 dataset to NCore V4 converter."""

from __future__ import annotations

import json
import logging
import os.path as osp
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import click
import numpy as np
import tqdm

from ncore.impl.common.transformations import HalfClosedInterval, se3_inverse, transform_bbox
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
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    PointCloudsComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
)
from ncore.impl.data.v4.types import ComponentGroupAssignments
from ncore.impl.data_converter.base import FileBasedDataConverter, FileBasedDataConverterConfig
from t4_devkit import Tier4
from t4_devkit.dataclass.pointcloud import PointCloudMetainfo, RadarPointCloud
from t4_devkit.schema import SchemaName
from t4_devkit.schema.tables.sample_data import FileFormat, SampleData
from tools.data_converter.cli import cli
from tools.data_converter.t4.pcd import ParsedPcd, load_pcd


CONVERTER_VERSION = "1.0.0"


def _sanitize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _channel_to_ncore_id(channel: str) -> str:
    parts = channel.lower().split("_")
    if not parts:
        raise ValueError(f"Invalid sensor channel: {channel!r}")

    modality = parts[0]
    suffix = "_".join(parts[1:])
    if suffix:
        return f"{modality}_{suffix}"
    return modality


def _quaternion_to_matrix(rotation: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation.rotation_matrix, dtype=np.float64)
    return matrix


def _pose_matrix(translation: Any, rotation: Any) -> np.ndarray:
    pose = _quaternion_to_matrix(rotation)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose


def _quaternion_to_xyz_euler(rotation: Any) -> tuple[float, float, float]:
    matrix = np.asarray(rotation.rotation_matrix, dtype=np.float64)
    sy = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    singular = sy < 1e-8

    if not singular:
        x = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        y = float(np.arctan2(-matrix[2, 0], sy))
        z = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        x = float(np.arctan2(-matrix[1, 2], matrix[1, 1]))
        y = float(np.arctan2(-matrix[2, 0], sy))
        z = 0.0

    return x, y, z


def _image_format_name(fileformat: FileFormat) -> str:
    return {
        FileFormat.JPG: "jpeg",
        FileFormat.PNG: "png",
    }[fileformat]


def _resolve_optional_path(data_root: str, relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    return osp.join(data_root, relative_path)


def _load_lidar_scan(file_path: str, metainfo_path: str | None) -> np.ndarray:
    num_pts_feats = 5
    if metainfo_path is not None and osp.exists(metainfo_path):
        num_pts_feats = PointCloudMetainfo.from_file(metainfo_path).num_pts_feats
    scan = np.fromfile(file_path, dtype=np.float32)
    if scan.size == 0:
        return np.zeros((0, num_pts_feats), dtype=np.float32)
    return scan.reshape((-1, num_pts_feats))


def _camera_model_from_t4(sample_data: SampleData, calibrated_sensor: Any) -> OpenCVPinholeCameraModelParameters:
    intrinsic = np.asarray(calibrated_sensor.camera_intrinsic, dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected 3x3 camera intrinsics, got {intrinsic.shape}")

    distortion = np.asarray(calibrated_sensor.camera_distortion, dtype=np.float32)

    radial = np.zeros(6, dtype=np.float32)
    tangential = np.zeros(2, dtype=np.float32)
    thin_prism = np.zeros(4, dtype=np.float32)

    if distortion.size >= 1:
        radial[0] = distortion[0]
    if distortion.size >= 2:
        radial[1] = distortion[1]
    if distortion.size >= 3:
        tangential[0] = distortion[2]
    if distortion.size >= 4:
        tangential[1] = distortion[3]
    if distortion.size >= 5:
        radial[2] = distortion[4]
    if distortion.size >= 8:
        radial[3:6] = distortion[5:8]
    if distortion.size >= 12:
        thin_prism[:] = distortion[8:12]

    return OpenCVPinholeCameraModelParameters(
        resolution=np.array([sample_data.width, sample_data.height], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        external_distortion_parameters=None,
        principal_point=np.array([intrinsic[0, 2], intrinsic[1, 2]], dtype=np.float32),
        focal_length=np.array([intrinsic[0, 0], intrinsic[1, 1]], dtype=np.float32),
        radial_coeffs=radial,
        tangential_coeffs=tangential,
        thin_prism_coeffs=thin_prism,
    )


@dataclass(kw_only=True, slots=True)
class T4Converter4Config(FileBasedDataConverterConfig):
    """Configuration for T4 to NCore V4 conversion."""

    revision: str | None = None
    scene_name: list[str] = field(default_factory=list)
    scene_token: list[str] = field(default_factory=list)
    store_type: Literal["itar", "directory"] = "itar"
    component_group_profile: Literal["default", "separate-sensors", "separate-all"] = "separate-sensors"
    store_sequence_meta: bool = True
    world_global_mode: Literal["none", "identity", "localized"] = "localized"
    pcd_map_path: str | None = None


class T4Converter4(FileBasedDataConverter):
    """Convert a T4 dataset loaded through ``t4-devkit`` into NCore V4."""

    def __init__(self, config: T4Converter4Config) -> None:
        super().__init__(config)
        self.revision = config.revision
        self.scene_name_filter = set(config.scene_name)
        self.scene_token_filter = set(config.scene_token)
        self.store_type = config.store_type
        self.component_group_profile = config.component_group_profile
        self.store_sequence_meta = config.store_sequence_meta
        self.world_global_mode = config.world_global_mode
        self.pcd_map_path = config.pcd_map_path
        self.logger = logging.getLogger(__name__)
        self.t4 = Tier4(str(self.root_dir), revision=self.revision, verbose=False)

    @staticmethod
    def get_sequence_ids(config: T4Converter4Config) -> list[str]:
        t4 = Tier4(str(config.root_dir), revision=config.revision, verbose=False)
        scenes = t4.scene

        if config.scene_name:
            scene_name_filter = set(config.scene_name)
            scenes = [scene for scene in scenes if scene.name in scene_name_filter]

        if config.scene_token:
            scene_token_filter = set(config.scene_token)
            scenes = [scene for scene in scenes if scene.token in scene_token_filter]

        return [scene.token for scene in scenes]

    @staticmethod
    def from_config(config: T4Converter4Config) -> T4Converter4:
        return T4Converter4(config)

    def _scene_samples(self, scene: Any) -> list[Any]:
        samples = []
        token = scene.first_sample_token
        while token:
            sample = self.t4.get(SchemaName.SAMPLE, token)
            samples.append(sample)
            token = sample.next
        return samples

    def _sample_data_in_scene(self, scene_samples: list[Any]) -> dict[str, list[Any]]:
        if not scene_samples:
            return {}

        scene_sample_tokens = {sample.token for sample in scene_samples}
        start_timestamp = scene_samples[0].timestamp
        end_timestamp = scene_samples[-1].timestamp

        by_channel: dict[str, list[Any]] = {}
        for sample_data in self.t4.sample_data:
            if not sample_data.is_valid:
                continue
            if sample_data.timestamp < start_timestamp or sample_data.timestamp > end_timestamp:
                continue
            if sample_data.sample_token and sample_data.sample_token not in scene_sample_tokens:
                continue

            by_channel.setdefault(sample_data.channel, []).append(sample_data)

        for records in by_channel.values():
            records.sort(key=lambda record: (record.timestamp, record.token))

        return by_channel

    def _sequence_id(self, scene: Any) -> str:
        return f"t4_{_sanitize_identifier(scene.name)}_{scene.token[:8]}"

    def _resolve_pcd_map_path(self) -> Path | None:
        """Resolve a PCD map file for the current scene, if one exists."""

        def _find_candidate(directory: Path, strict: bool) -> Path | None:
            if not directory.exists() or not directory.is_dir():
                return None

            pcd_files = sorted(directory.rglob("*.pcd"))
            if not pcd_files:
                return None

            preferred = [
                p
                for p in pcd_files
                if any(token in p.name.lower() for token in ("pcd_map", "pointcloud_map", "point_cloud_map", "map"))
            ]
            if len(preferred) == 1:
                return preferred[0]
            if len(preferred) > 1:
                if not strict:
                    return None
                raise ValueError(f"Ambiguous PCD map candidates in {directory}: {preferred}")
            if len(pcd_files) == 1:
                return pcd_files[0]
            if not strict:
                return None
            raise ValueError(f"Ambiguous PCD map candidates in {directory}: {pcd_files}")

        if self.pcd_map_path:
            configured = Path(self.pcd_map_path)
            if configured.exists():
                if configured.is_file():
                    return configured
                resolved = _find_candidate(configured, strict=True)
                if resolved is None:
                    raise FileNotFoundError(f"No PCD files found under {configured}")
                return resolved

            root_candidate = Path(str(self.root_dir)) / self.pcd_map_path
            if root_candidate.exists():
                if root_candidate.is_file():
                    return root_candidate
                resolved = _find_candidate(root_candidate, strict=True)
                if resolved is None:
                    raise FileNotFoundError(f"No PCD files found under {root_candidate}")
                return resolved

            raise FileNotFoundError(f"PCD map path does not exist: {self.pcd_map_path}")

        root_dir = Path(str(self.root_dir))
        relative_map_dir = root_dir / "annotation_dataset" / str(self.t4.dataset_id) / str(self.t4.version) / "map"
        resolved = _find_candidate(relative_map_dir, strict=False)
        if resolved is not None:
            return resolved
        return None

    def _store_pcd_map(self, sequence_start_timestamp_us: int, pcd_map_path: Path, reference_frame_id: str) -> None:
        try:
            parsed_pcd: ParsedPcd = load_pcd(pcd_map_path)
        except NotImplementedError as exc:
            self.logger.warning("Skipping PCD map export for %s: %s", pcd_map_path, exc)
            return
        attribute_schemas = {
            field.name: PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=field.values.dtype,
                shape_suffix=field.values.shape[1:],
            )
            for field in parsed_pcd.fields
        }
        point_cloud_writer = self.store_writer.register_component_writer(
            PointCloudsComponent.Writer,
            component_instance_name="map",
            group_name=self.component_groups.point_clouds_component_groups.get("map"),
            coordinate_unit=PointCloud.CoordinateUnit.METERS,
            attribute_schemas=attribute_schemas,
        )
        point_cloud_writer.store_pc(
            xyz=parsed_pcd.xyz,
            reference_frame_id=reference_frame_id,
            reference_frame_timestamp_us=sequence_start_timestamp_us,
            attributes={field.name: field.values for field in parsed_pcd.fields},
            generic_meta_data={
                "source_type": "pcd_map",
                "source_path": str(pcd_map_path),
                "pcd_header": parsed_pcd.header,
            },
        )

    def _store_poses(self, ego_pose_records: list[Any]) -> np.ndarray | None:
        timestamps_us = np.array([record.timestamp for record in ego_pose_records], dtype=np.uint64)
        poses = np.stack(
            [_pose_matrix(record.translation, record.rotation) for record in ego_pose_records],
            axis=0,
        )
        T_world_world_global = None

        match self.world_global_mode:
            case "none":
                pass
            case "identity":
                T_world_world_global = np.eye(4, dtype=np.float64)
            case "localized":
                T_world_world_global = poses[0].copy()
                poses = se3_inverse(T_world_world_global)[None] @ poses
            case _:
                raise ValueError(f"Unknown world_global_mode: {self.world_global_mode!r}")

        self.poses_writer.store_dynamic_pose(
            source_frame_id="rig",
            target_frame_id="world",
            poses=poses.astype(np.float32),
            timestamps_us=timestamps_us,
        )

        if T_world_world_global is not None:
            self.poses_writer.store_static_pose(
                source_frame_id="world",
                target_frame_id="world_global",
                pose=T_world_world_global,
            )

        return T_world_world_global

    def _store_cameras(
        self,
        scene_sample_data: dict[str, list[Any]],
        calibrated_sensors_by_token: dict[str, Any],
        active_camera_ids: list[str],
        camera_id_by_channel: dict[str, str],
    ) -> None:
        for channel, records in scene_sample_data.items():
            if channel not in camera_id_by_channel:
                continue
            if not records:
                continue

            sample_data0 = records[0]
            calibrated_sensor = calibrated_sensors_by_token[sample_data0.calibrated_sensor_token]
            sensor_id = camera_id_by_channel[channel]
            if sensor_id not in active_camera_ids:
                continue

            self.intrinsics_writer.store_camera_intrinsics(
                camera_id=sensor_id,
                camera_model_parameters=_camera_model_from_t4(sample_data0, calibrated_sensor),
            )
            self.poses_writer.store_static_pose(
                source_frame_id=sensor_id,
                target_frame_id="rig",
                pose=_pose_matrix(calibrated_sensor.translation, calibrated_sensor.rotation).astype(np.float32),
            )

            camera_writer = self.store_writer.register_component_writer(
                CameraSensorComponent.Writer,
                component_instance_name=sensor_id,
                group_name=self.component_groups.camera_component_groups.get(sensor_id),
            )

            for sample_data in tqdm.tqdm(records, desc=f"Process {sensor_id}"):
                image_path = self.t4.get_sample_data_path(sample_data.token)
                with open(image_path, "rb") as f:
                    image_binary = f.read()

                timestamp = int(sample_data.timestamp)
                camera_writer.store_frame(
                    image_binary_data=image_binary,
                    image_format=_image_format_name(sample_data.fileformat),
                    frame_timestamps_us=np.array([timestamp, timestamp], dtype=np.uint64),
                    generic_data={},
                    generic_meta_data={},
                )

    def _store_lidar_point_clouds(
        self,
        scene_sample_data: dict[str, list[Any]],
        calibrated_sensors_by_token: dict[str, Any],
        active_lidar_ids: list[str],
        lidar_id_by_channel: dict[str, str],
    ) -> None:
        for channel, records in scene_sample_data.items():
            if channel not in lidar_id_by_channel:
                continue

            sensor_id = lidar_id_by_channel[channel]
            if sensor_id not in active_lidar_ids:
                continue

            sample_data0 = records[0]
            calibrated_sensor = calibrated_sensors_by_token[sample_data0.calibrated_sensor_token]
            self.poses_writer.store_static_pose(
                source_frame_id=sensor_id,
                target_frame_id="rig",
                pose=_pose_matrix(calibrated_sensor.translation, calibrated_sensor.rotation).astype(np.float32),
            )

            first_scan = _load_lidar_scan(
                self.t4.get_sample_data_path(sample_data0.token),
                _resolve_optional_path(self.t4.data_root, sample_data0.info_filename),
            )
            has_ring_index = first_scan.shape[1] >= 5

            attribute_schemas = {
                "intensity": PointCloudsComponent.AttributeSchema(
                    transform_type=PointCloud.AttributeTransformType.INVARIANT,
                    dtype=np.dtype("float32"),
                ),
            }
            if has_ring_index:
                attribute_schemas["ring_index"] = PointCloudsComponent.AttributeSchema(
                    transform_type=PointCloud.AttributeTransformType.INVARIANT,
                    dtype=np.dtype("int32"),
                )

            point_cloud_writer = self.store_writer.register_component_writer(
                PointCloudsComponent.Writer,
                component_instance_name=sensor_id,
                group_name=self.component_groups.point_clouds_component_groups.get(sensor_id),
                coordinate_unit=PointCloud.CoordinateUnit.METERS,
                attribute_schemas=attribute_schemas,
            )

            for sample_data in tqdm.tqdm(records, desc=f"Process {sensor_id}"):
                scan = _load_lidar_scan(
                    self.t4.get_sample_data_path(sample_data.token),
                    _resolve_optional_path(self.t4.data_root, sample_data.info_filename),
                )
                xyz = scan[:, :3].astype(np.float32)
                attributes: dict[str, np.ndarray] = {
                    "intensity": scan[:, 3].astype(np.float32) if scan.shape[1] >= 4 else np.zeros(len(xyz), np.float32)
                }
                if has_ring_index:
                    if scan.shape[1] >= 5:
                        attributes["ring_index"] = scan[:, 4].astype(np.int32)
                    else:
                        attributes["ring_index"] = np.full(len(xyz), -1, dtype=np.int32)

                point_cloud_writer.store_pc(
                    xyz=xyz,
                    reference_frame_id="rig",
                    reference_frame_timestamp_us=int(sample_data.timestamp),
                    attributes=attributes,
                    generic_meta_data={
                        "source_channel": channel,
                        "t4_fileformat": sample_data.fileformat.value,
                    },
                )

    def _store_radar_point_clouds(
        self,
        scene_sample_data: dict[str, list[Any]],
        calibrated_sensors_by_token: dict[str, Any],
        active_radar_ids: list[str],
        radar_id_by_channel: dict[str, str],
    ) -> None:
        radar_attribute_schemas = {
            "dyn_prop": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
            ),
            "point_id": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint16"),
            ),
            "rcs": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("float32"),
            ),
            "vx": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("float32"),
            ),
            "vy": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("float32"),
            ),
            "vx_comp": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("float32"),
            ),
            "vy_comp": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("float32"),
            ),
            "is_quality_valid": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
            ),
            "ambig_state": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
            ),
            "x_rms": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
            ),
            "y_rms": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
            ),
            "invalid_state": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
            ),
            "pdh0": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
            ),
            "vx_rms": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
            ),
            "vy_rms": PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT,
                dtype=np.dtype("uint8"),
            ),
        }

        for channel, records in scene_sample_data.items():
            if channel not in radar_id_by_channel:
                continue

            sensor_id = radar_id_by_channel[channel]
            if sensor_id not in active_radar_ids:
                continue

            sample_data0 = records[0]
            calibrated_sensor = calibrated_sensors_by_token[sample_data0.calibrated_sensor_token]
            self.poses_writer.store_static_pose(
                source_frame_id=sensor_id,
                target_frame_id="rig",
                pose=_pose_matrix(calibrated_sensor.translation, calibrated_sensor.rotation).astype(np.float32),
            )

            point_cloud_writer = self.store_writer.register_component_writer(
                PointCloudsComponent.Writer,
                component_instance_name=sensor_id,
                group_name=self.component_groups.point_clouds_component_groups.get(sensor_id),
                coordinate_unit=PointCloud.CoordinateUnit.METERS,
                attribute_schemas=radar_attribute_schemas,
            )

            for sample_data in tqdm.tqdm(records, desc=f"Process {sensor_id}"):
                point_cloud = RadarPointCloud.from_file(self.t4.get_sample_data_path(sample_data.token))
                points = point_cloud.points
                xyz = points[:3, :].T.astype(np.float32)
                attributes = {
                    "dyn_prop": points[3, :].astype(np.uint8),
                    "point_id": points[4, :].astype(np.uint16),
                    "rcs": points[5, :].astype(np.float32),
                    "vx": points[6, :].astype(np.float32),
                    "vy": points[7, :].astype(np.float32),
                    "vx_comp": points[8, :].astype(np.float32),
                    "vy_comp": points[9, :].astype(np.float32),
                    "is_quality_valid": points[10, :].astype(np.uint8),
                    "ambig_state": points[11, :].astype(np.uint8),
                    "x_rms": points[12, :].astype(np.uint8),
                    "y_rms": points[13, :].astype(np.uint8),
                    "invalid_state": points[14, :].astype(np.uint8),
                    "pdh0": points[15, :].astype(np.uint8),
                    "vx_rms": points[16, :].astype(np.uint8),
                    "vy_rms": points[17, :].astype(np.uint8),
                }
                point_cloud_writer.store_pc(
                    xyz=xyz,
                    reference_frame_id=sensor_id,
                    reference_frame_timestamp_us=int(sample_data.timestamp),
                    attributes=attributes,
                    generic_meta_data={
                        "source_channel": channel,
                        "t4_fileformat": sample_data.fileformat.value,
                    },
                )

    def _store_cuboids(self, scene_samples: list[Any], T_world_world_global: np.ndarray | None) -> None:
        observations: list[CuboidTrackObservation] = []
        T_world_global_world = se3_inverse(T_world_world_global) if T_world_world_global is not None else None

        for sample in scene_samples:
            for annotation_token in sample.ann_3ds:
                annotation = self.t4.get(SchemaName.SAMPLE_ANNOTATION, annotation_token)
                instance = self.t4.get(SchemaName.INSTANCE, annotation.instance_token)

                bbox = BBox3(
                    centroid=tuple(float(v) for v in annotation.translation),
                    dim=tuple(float(v) for v in annotation.size),
                    rot=_quaternion_to_xyz_euler(annotation.rotation),
                )
                if T_world_global_world is not None:
                    bbox = BBox3.from_array(transform_bbox(bbox.to_array(), T_world_global_world))

                source = LabelSource.AUTOLABEL if annotation.automatic_annotation else LabelSource.GT_ANNOTATION
                source_version = None
                if annotation.autolabel_metadata:
                    source_version = ",".join(model.name for model in annotation.autolabel_metadata)

                observations.append(
                    CuboidTrackObservation(
                        track_id=instance.instance_name or instance.token,
                        class_id=annotation.category_name,
                        timestamp_us=int(sample.timestamp),
                        reference_frame_id="world",
                        reference_frame_timestamp_us=int(sample.timestamp),
                        bbox3=bbox,
                        source=source,
                        source_version=source_version,
                    )
                )

        self.store_writer.register_component_writer(
            CuboidsComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.cuboid_track_observations_component_group,
        ).store_observations(observations)

    def convert_sequence(self, sequence_id: str) -> None:
        scene = self.t4.get(SchemaName.SCENE, sequence_id)
        scene_samples = self._scene_samples(scene)
        if not scene_samples:
            raise ValueError(f"Scene {scene.name} ({scene.token}) does not contain any samples")

        scene_sample_data = self._sample_data_in_scene(scene_samples)
        sequence_id_str = self._sequence_id(scene)
        self.logger.info("Converting T4 scene %s (%s) -> %s", scene.name, scene.token, sequence_id_str)

        sequence_timestamp_interval_us = HalfClosedInterval.from_start_end(
            scene_samples[0].timestamp,
            scene_samples[-1].timestamp,
        )

        ego_pose_tokens = sorted(
            {record.ego_pose_token for records in scene_sample_data.values() for record in records}
            | {self.t4.get(SchemaName.SAMPLE_DATA, sample.data[channel]).ego_pose_token for sample in scene_samples for channel in sample.data},
            key=lambda token: self.t4.get(SchemaName.EGO_POSE, token).timestamp,
        )
        ego_pose_records = [self.t4.get(SchemaName.EGO_POSE, token) for token in ego_pose_tokens]

        camera_channels = sorted(
            channel
            for channel, records in scene_sample_data.items()
            if records and records[0].modality.value == "camera"
        )
        lidar_channels = sorted(
            channel
            for channel, records in scene_sample_data.items()
            if records and records[0].modality.value == "lidar"
        )
        radar_channels = sorted(
            channel
            for channel, records in scene_sample_data.items()
            if records and records[0].modality.value == "radar"
        )

        camera_id_by_channel = {channel: _channel_to_ncore_id(channel) for channel in camera_channels}
        lidar_id_by_channel = {channel: _channel_to_ncore_id(channel) for channel in lidar_channels}
        radar_id_by_channel = {channel: _channel_to_ncore_id(channel) for channel in radar_channels}

        pcd_map_path = self._resolve_pcd_map_path()

        active_camera_ids = self.get_active_camera_ids(list(camera_id_by_channel.values()))
        active_lidar_ids = self.get_active_lidar_ids(list(lidar_id_by_channel.values()))
        active_radar_ids = self.get_active_radar_ids(list(radar_id_by_channel.values()))
        point_cloud_ids = active_lidar_ids + active_radar_ids + (["map"] if pcd_map_path is not None else [])

        self.component_groups = ComponentGroupAssignments.create(
            camera_ids=active_camera_ids,
            lidar_ids=[],
            radar_ids=[],
            point_clouds_ids=point_cloud_ids,
            profile=self.component_group_profile,
        )

        self.store_writer = SequenceComponentGroupsWriter(
            output_dir_path=self.output_dir / sequence_id_str,
            store_base_name=sequence_id_str,
            sequence_id=sequence_id_str,
            sequence_timestamp_interval_us=sequence_timestamp_interval_us,
            store_type=self.store_type,
            generic_meta_data={
                "dataset_id": self.t4.dataset_id,
                "dataset_version": self.t4.version,
                "scene_name": scene.name,
                "scene_token": scene.token,
                "log_token": scene.log_token,
                "converter": "t4-v4",
                "converter_version": CONVERTER_VERSION,
            },
        )
        self.poses_writer = self.store_writer.register_component_writer(
            PosesComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.poses_component_group,
            generic_meta_data={
                "calibration_type": "t4:calib",
                "egomotion_type": "t4:ego_pose",
            },
        )
        self.intrinsics_writer = self.store_writer.register_component_writer(
            IntrinsicsComponent.Writer,
            component_instance_name="default",
            group_name=self.component_groups.intrinsics_component_group,
        )

        calibrated_sensors_by_token = {
            record.token: record for record in self.t4.calibrated_sensor
        }

        T_world_world_global = self._store_poses(ego_pose_records)
        self._store_cameras(scene_sample_data, calibrated_sensors_by_token, active_camera_ids, camera_id_by_channel)
        self._store_lidar_point_clouds(
            scene_sample_data,
            calibrated_sensors_by_token,
            active_lidar_ids,
            lidar_id_by_channel,
        )
        self._store_radar_point_clouds(
            scene_sample_data,
            calibrated_sensors_by_token,
            active_radar_ids,
            radar_id_by_channel,
        )
        if pcd_map_path is not None:
            self._store_pcd_map(
                scene_samples[0].timestamp,
                pcd_map_path,
                reference_frame_id="world_global" if T_world_world_global is not None else "world",
            )
        self._store_cuboids(scene_samples, T_world_world_global)

        ncore_4_paths = self.store_writer.finalize()

        if self.store_sequence_meta:
            sequence_component_reader = SequenceComponentGroupsReader(ncore_4_paths)
            sequence_meta_path = self.output_dir / sequence_id_str / f"{sequence_component_reader.sequence_id}.json"
            with sequence_meta_path.open("w") as f:
                json.dump(sequence_component_reader.get_sequence_meta().to_dict(), f, indent=2)
            self.logger.info("Wrote sequence meta data %s", sequence_meta_path)


@cli.command("t4-v4")
@click.option(
    "--revision",
    type=str,
    default=None,
    help="T4 dataset revision/version to load. If omitted, t4-devkit loads the latest version.",
)
@click.option(
    "--scene-name",
    multiple=True,
    type=str,
    default=[],
    help="Specific scene name(s) to convert. If omitted, converts all scenes.",
)
@click.option(
    "--scene-token",
    multiple=True,
    type=str,
    default=[],
    help="Specific scene token(s) to convert. If omitted, converts all scenes.",
)
@click.option(
    "--store-type",
    type=click.Choice(["itar", "directory"], case_sensitive=False),
    default="itar",
    show_default=True,
    help="Output store type",
)
@click.option(
    "component_group_profile",
    "--profile",
    type=click.Choice(["default", "separate-sensors", "separate-all"], case_sensitive=False),
    default="separate-sensors",
    show_default=True,
    help="""Output profile, one of:
        - "default": All components defaults or overrides
        - "separate-sensors": Each sensor gets its own group named "<sensor_id>", remaining components use overrides
        - "separate-all": Each component type gets its own group named after the component type, e.g. "poses", "intrinsics", respecting overwrites if provided""",
)
@click.option(
    "store_sequence_meta", "--sequence-meta/--no-sequence-meta", default=True, help="Generate sequence meta-data?"
)
@click.option(
    "--world-global-mode",
    type=click.Choice(["none", "identity", "localized"], case_sensitive=False),
    default="localized",
    show_default=True,
    help="""Controls whether a ("world", "world_global") static pose is stored:
        - "none": No world_global pose. Poses remain in source coordinates.
        - "identity": Store an identity world_global pose. Poses remain in source coordinates.
        - "localized": Rebase poses relative to the first frame and store the original first pose as world->world_global.""",
)
@click.option(
    "--pcd-map-path",
    type=str,
    default=None,
    help="Optional path to a PCD map file or directory. If omitted, the converter tries to auto-discover a unique .pcd map under the T4 root.",
)
@click.pass_context
def t4_v4(ctx, *_, **kwargs):
    """T4-specific data conversion (V4 format)."""

    config = T4Converter4Config(**{**vars(ctx.obj), **kwargs})
    T4Converter4.convert(config)


if __name__ == "__main__":
    cli(show_default=True)
