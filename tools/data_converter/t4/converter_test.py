# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the T4 converter on a synthetic T4 sequence with gaussian_factory artifacts."""

import io
import json
import tempfile
import unittest

from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np

from PIL import Image as PILImage
from upath import UPath

from ncore.impl.data.types import OpenCVPinholeCameraModelParameters
from ncore.impl.data.v4.components import (
    CameraLabelsComponent,
    CameraSensorComponent,
    CuboidsComponent,
    IntrinsicsComponent,
    LidarSensorComponent,
    PointCloudsComponent,
    PosesComponent,
    SequenceComponentGroupsReader,
)
from tools.data_converter.t4.converter import T4Converter4, T4Converter4Config
from tools.data_converter.t4.gaussian_factory import DEPTH_UINT16_SCALE_M
from tools.data_converter.t4.utils import t4_distortion_to_opencv


W, H = 16, 12
CAMERAS = ["CAM_FRONT", "CAM_BACK"]
N_SAMPLES = 3
T0 = 1_700_000_000_000_000
DISTORTION = [0.1, -0.2, 0.001, -0.002, 0.3, 0.05, -0.06, 0.07]


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    PILImage.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(buf, format="jpeg", quality=95)
    return buf.getvalue()


def make_t4_sequence(root: Path) -> Dict[str, Any]:
    """Writes a minimal T4 sequence: 3 keyframe samples + 1 non-keyframe sweep per channel."""
    rng = np.random.default_rng(0)
    sensors = [{"token": f"s_{ch}", "channel": ch, "modality": "camera"} for ch in CAMERAS]
    sensors.append({"token": "s_lidar", "channel": "LIDAR_CONCAT", "modality": "lidar"})
    calibs = [
        {
            "token": f"c_{ch}",
            "sensor_token": f"s_{ch}",
            "translation": [1.0, 0.0, 1.5],
            "rotation": [0.5, -0.5, 0.5, -0.5],
            "camera_intrinsic": [[10.0, 0, 8.0], [0, 11.0, 6.0], [0, 0, 1]],
            "camera_distortion": DISTORTION,
        }
        for ch in CAMERAS
    ]
    calibs.append(
        {
            "token": "c_LIDAR_CONCAT",
            "sensor_token": "s_lidar",
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "camera_intrinsic": [],
            "camera_distortion": [],
        }
    )
    samples, sample_data, ego_poses = [], [], []
    lidar_points: Dict[str, np.ndarray] = {}
    for i in range(N_SAMPLES):
        samples.append(
            {
                "token": f"sample{i}",
                "timestamp": T0 + i * 100_000,
                "scene_token": "scene0",
                "next": f"sample{i + 1}" if i + 1 < N_SAMPLES else "",
                "prev": f"sample{i - 1}" if i else "",
            }
        )
    frame_no = 0
    for i in range(N_SAMPLES):
        # keyframe at t, non-keyframe sweep at t + 50 ms
        for is_key, dt in ((True, 0), (False, 50_000)):
            if not is_key and i == N_SAMPLES - 1:
                continue
            ts = T0 + i * 100_000 + dt
            ego_poses.append(
                {
                    "token": f"ego{frame_no}",
                    "timestamp": ts,
                    "translation": [ts * 1e-6 - T0 * 1e-6, 0, 0],
                    "rotation": [1, 0, 0, 0],
                }
            )
            for ch in CAMERAS + ["LIDAR_CONCAT"]:
                is_cam = ch != "LIDAR_CONCAT"
                fn = f"data/{ch}/{frame_no:05d}.{'jpg' if is_cam else 'pcd.bin'}"
                (root / fn).parent.mkdir(parents=True, exist_ok=True)
                if is_cam:
                    (root / fn).write_bytes(_jpeg(frame_no))
                else:
                    pts = rng.normal(0, 10, (50, 5)).astype(np.float32)
                    pts[:, 3] = rng.integers(0, 256, 50)
                    pts[:, 4] = rng.integers(-1, 32, 50)  # -1 = unknown ring
                    pts[0, :3] = 0.0  # zero-range point is dropped
                    pts.tofile(root / fn)
                    lidar_points[fn] = pts
                sample_data.append(
                    {
                        "token": f"sd_{ch}_{frame_no}",
                        "sample_token": f"sample{i}",
                        "ego_pose_token": f"ego{frame_no}",
                        "calibrated_sensor_token": f"c_{ch}",
                        "filename": fn,
                        "fileformat": "jpg" if is_cam else "pcd.bin",
                        "width": W if is_cam else 0,
                        "height": H if is_cam else 0,
                        "timestamp": ts + (1000 if is_cam else 0),
                        "is_key_frame": is_key,
                        "next": "",
                        "prev": "",
                    }
                )
            frame_no += 1
    # ego_pose starts before the first sample_data (as in the tier4_perception_dataset sample)
    ego_poses.insert(
        0, {"token": "ego_early", "timestamp": T0 - 30_000, "translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}
    )

    ann = root / "annotation"
    _write_json(
        ann / "scene.json",
        [{"token": "scene0", "name": "synthetic", "log_token": "log0", "first_sample_token": "sample0"}],
    )
    _write_json(ann / "sensor.json", sensors)
    _write_json(ann / "calibrated_sensor.json", calibs)
    _write_json(ann / "ego_pose.json", ego_poses)
    _write_json(ann / "sample.json", samples)
    _write_json(ann / "sample_data.json", sample_data)
    _write_json(ann / "category.json", [{"token": "cat0", "name": "vehicle.car"}])
    _write_json(ann / "instance.json", [{"token": "inst0", "category_token": "cat0"}])
    _write_json(ann / "visibility.json", [{"token": "v4", "level": "full"}, {"token": "v1", "level": "none"}])
    _write_json(
        ann / "sample_annotation.json",
        [
            {
                "token": f"ann{i}",
                "sample_token": f"sample{i}",
                "instance_token": "inst0",
                "visibility_token": "v4" if i else "v1",
                "translation": [5.0 + i, 1.0, 0.5],
                "size": [2.0, 4.5, 1.5],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "num_lidar_pts": 10 * i,
                "num_radar_pts": 0,
            }
            for i in range(N_SAMPLES)
        ],
    )
    (root / "map").mkdir(exist_ok=True)
    (root / "map" / "lanelet2_map.osm").write_text("<osm/>")
    return {"sample_data": sample_data, "lidar_points": lidar_points}


def make_gf_outputs(gf: Path, sample_data: List[dict]) -> Dict[str, Any]:
    """Writes gaussian_factory artifacts for the keyframe camera frames."""
    rng = np.random.default_rng(1)
    expected: Dict[str, Any] = {}
    poses = {}
    keyframes = [sd for sd in sample_data if sd["is_key_frame"] and not sd["filename"].startswith("data/LIDAR")]
    for sd in keyframes:
        ch = sd["filename"].split("/")[1]
        stem = Path(sd["filename"]).stem
        kf = int(sd["sample_token"].removeprefix("sample"))
        inst = rng.integers(0, 3, (H, W), dtype=np.uint8)
        sky = (rng.random((H, W)) > 0.5).astype(np.uint8) * 255
        depth = np.where(rng.random((H, W)) > 0.7, rng.uniform(1, 80, (H, W)), 0).astype(np.float32)
        ma_depth = rng.uniform(0.3, 20, (H // 2, W // 2)).astype(np.float32)
        for sub, arr in (("sam_masks", inst), ("sky_masks", sky)):
            (gf / sub / ch).mkdir(parents=True, exist_ok=True)
            PILImage.fromarray(arr, "L").save(gf / sub / ch / f"{stem}.png")
        (gf / "lidar_depth_accum" / ch).mkdir(parents=True, exist_ok=True)
        np.save(gf / "lidar_depth_accum" / ch / f"frame_{kf:04d}.npy", depth)
        (gf / "ma" / "view_depth").mkdir(parents=True, exist_ok=True)
        np.save(gf / "ma" / "view_depth" / f"{sd['token']}.npy", ma_depth)
        c2w = np.eye(4)
        c2w[:3, 3] = rng.normal(size=3)
        poses[sd["token"]] = c2w.tolist()
        expected[sd["token"]] = {"instance": inst, "sky": sky, "lidar_depth": depth, "ma": ma_depth, "c2w": c2w}
    _write_json(gf / "sam_masks" / "track_id_mapping.json", {"inst0": 1})
    _write_json(gf / "refined_poses_step000100.json", {"meta": {"step": 100}, "poses": poses})
    return expected


def _config(t4: Path, out: Path, **kwargs: Any) -> T4Converter4Config:
    base: Dict[str, Any] = dict(
        root_dir=str(t4),
        output_dir=str(out),
        no_cameras=False,
        camera_ids=None,
        no_lidars=False,
        lidar_ids=None,
        no_radars=False,
        radar_ids=None,
        verbose=False,
        debug=False,
        debug_port=0,
        store_type="directory",
    )
    base.update(kwargs)
    return T4Converter4Config(**base)


def _open(out: Path) -> SequenceComponentGroupsReader:
    seq_dir = out / "t4seq"  # not a webauto <id>/<version> dir, so the name is kept
    return SequenceComponentGroupsReader([UPath(p) for p in sorted(seq_dir.glob("*.zarr"))])


class TestT4Converter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.t4 = cls.tmp / "t4seq"
        info = make_t4_sequence(cls.t4)
        cls.sample_data = info["sample_data"]
        cls.lidar_points = info["lidar_points"]
        cls.gf = cls.tmp / "gf"
        cls.expected = make_gf_outputs(cls.gf, cls.sample_data)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_distortion_mapping(self) -> None:
        radial, tangential, thin_prism = t4_distortion_to_opencv(DISTORTION)
        np.testing.assert_allclose(radial, [0.1, -0.2, 0.3, 0.05, -0.06, 0.07], rtol=1e-6)
        np.testing.assert_allclose(tangential, [0.001, -0.002], rtol=1e-6)
        np.testing.assert_array_equal(thin_prism, np.zeros(4))
        radial5, _, _ = t4_distortion_to_opencv([0.1, -0.2, 0.0, 0.0, 0.3])
        np.testing.assert_allclose(radial5, [0.1, -0.2, 0.3, 0, 0, 0], rtol=1e-6)

    def test_all_frames_ray_bundle(self) -> None:
        out = self.tmp / "out_all"
        T4Converter4.convert(_config(self.t4, out))
        reader = _open(out)

        cams = reader.open_component_readers(CameraSensorComponent.Reader)
        self.assertEqual(sorted(cams), sorted(CAMERAS))
        self.assertEqual(cams["CAM_FRONT"].frames_count, 2 * N_SAMPLES - 1)

        intr = reader.open_component_readers(IntrinsicsComponent.Reader)["default"]
        params = intr.get_camera_model_parameters("CAM_FRONT")
        assert isinstance(params, OpenCVPinholeCameraModelParameters)
        np.testing.assert_allclose(params.radial_coeffs, [0.1, -0.2, 0.3, 0.05, -0.06, 0.07], rtol=1e-6)

        lidar = reader.open_component_readers(LidarSensorComponent.Reader)["LIDAR_CONCAT"]
        ts = int(lidar.frames_timestamps_us[0, 1])
        pts = self.lidar_points["data/LIDAR_CONCAT/00000.pcd.bin"][1:]
        direction = lidar.get_frame_ray_bundle_data(ts, "direction")
        distance = lidar.get_frame_ray_bundle_return_data(ts, "distance_m", 0)
        np.testing.assert_allclose(direction * distance[:, None], pts[:, :3], atol=1e-4)
        np.testing.assert_array_equal(lidar.get_frame_generic_data(ts, "ring"), pts[:, 4].astype(np.int16))
        meta = lidar.get_frame_generic_meta_data(ts)
        self.assertEqual(meta["t4_sample_data_token"], "sd_LIDAR_CONCAT_0")
        self.assertEqual(meta["t4_keyframe_index"], 0)

        poses = reader.open_component_readers(PosesComponent.Reader)["default"]
        self.assertIn("t4_lanelet2_map_osm", poses.get_generic_data_names())
        self.assertEqual(bytes(poses.get_generic_data("t4_lanelet2_map_osm")), b"<osm/>")

        cuboids = reader.open_component_readers(CuboidsComponent.Reader)["default"]
        self.assertEqual(len(list(cuboids.get_observations())), N_SAMPLES)
        np.testing.assert_array_equal(cuboids.get_generic_data("t4_num_lidar_pts"), [0, 10, 20])
        levels = cast(List[str], cuboids.generic_meta_data["t4_visibility_levels"])
        self.assertEqual([levels[i] for i in cuboids.get_generic_data("t4_visibility_index")], ["none", "full", "full"])

    def test_gaussian_factory_keyframes_point_cloud(self) -> None:
        out = self.tmp / "out_gf"
        T4Converter4.convert(
            _config(
                self.t4,
                out,
                keyframes_only=True,
                lidar_format="point-cloud",
                gf_root=str(self.gf),
                gf_layer_dirs=(f"mapanything_depth={self.gf / 'ma' / 'view_depth'}",),
                gf_camera_poses_json=str(self.gf / "refined_poses_step000100.json"),
            )
        )
        reader = _open(out)
        cams = reader.open_component_readers(CameraSensorComponent.Reader)
        self.assertEqual(cams["CAM_FRONT"].frames_count, N_SAMPLES)

        pcs = reader.open_component_readers(PointCloudsComponent.Reader)["LIDAR_CONCAT"]
        self.assertEqual(pcs.pcs_count, N_SAMPLES)
        pts = self.lidar_points["data/LIDAR_CONCAT/00000.pcd.bin"][1:]
        np.testing.assert_array_equal(pcs.get_pc_xyz(0), pts[:, :3])
        np.testing.assert_array_equal(pcs.get_pc_attribute(0, "intensity"), pts[:, 3].astype(np.uint8))
        np.testing.assert_array_equal(pcs.get_pc_attribute(0, "ring"), pts[:, 4].astype(np.int16))

        labels = reader.open_component_readers(CameraLabelsComponent.Reader)
        self.assertEqual(
            sorted(labels),
            sorted(
                f"{k}@{c}"
                for k in ("segmentation.instance", "mask.sky", "depth.z_lidar_accum", "depth.z_mapanything")
                for c in CAMERAS
            ),
        )
        self.assertEqual(labels["segmentation.instance@CAM_FRONT"].generic_meta_data["track_id_mapping"], {"inst0": 1})
        for name, lr in labels.items():
            self.assertEqual(lr.labels_count, N_SAMPLES, name)
            for ts in lr.timestamps_us:
                handle = lr.get_label(int(ts))
                exp = self.expected[str(handle.generic_meta_data["t4_sample_data_token"])]
                data = handle.get_data()
                if name.startswith("segmentation.instance"):
                    np.testing.assert_array_equal(data, exp["instance"])
                elif name.startswith("mask.sky"):
                    np.testing.assert_array_equal(data, exp["sky"])
                elif name.startswith("depth.z_lidar_accum"):
                    np.testing.assert_allclose(data, exp["lidar_depth"], atol=DEPTH_UINT16_SCALE_M / 2 + 1e-6)
                    np.testing.assert_array_equal(data == 0, exp["lidar_depth"] == 0)
                else:
                    np.testing.assert_allclose(data, exp["ma"], atol=DEPTH_UINT16_SCALE_M / 2 + 1e-6)

        gf_poses = reader.open_component_readers(PosesComponent.Reader)["gaussian_factory"]
        poses, ts = gf_poses.get_dynamic_pose("CAM_FRONT", "world")
        self.assertEqual(len(ts), N_SAMPLES)
        np.testing.assert_allclose(poses[0], self.expected["sd_CAM_FRONT_0"]["c2w"])

    def test_gaussian_factory_float32_depth(self) -> None:
        out = self.tmp / "out_gf_f32"
        T4Converter4.convert(
            _config(
                self.t4,
                out,
                keyframes_only=True,
                gf_root=str(self.gf),
                gf_layers=("lidar_depth",),
                gf_depth_encoding="float32",
            )
        )
        labels = _open(out).open_component_readers(CameraLabelsComponent.Reader)
        self.assertEqual(sorted(labels), [f"depth.z_lidar_accum@{c}" for c in sorted(CAMERAS)])
        lr = labels["depth.z_lidar_accum@CAM_BACK"]
        handle = lr.get_label(int(lr.timestamps_us[0]))
        exp = self.expected[str(handle.generic_meta_data["t4_sample_data_token"])]
        np.testing.assert_array_equal(handle.get_data(), exp["lidar_depth"])

    def test_rebuilt_lidar_keyframes_only(self) -> None:
        # lidar_rebuild names frames by the ordinal among *all* lidar frames
        rebuilt = self.tmp / "rebuilt"
        rebuilt.mkdir()
        lidar_sds = sorted(
            (sd for sd in self.sample_data if sd["filename"].startswith("data/LIDAR")), key=lambda sd: sd["timestamp"]
        )
        for i, _ in enumerate(lidar_sds):
            rec = np.zeros((4, 7), dtype=np.float32)
            rec[:, 0] = 1.0 + i  # x encodes the ordinal
            rec[:, 3] = 100
            rec[:, 4] = np.arange(4)
            rec[:, 6] = np.array([0.0, 0.01, 0.02, 0.03]) + 1e-7
            rec.tofile(rebuilt / f"{i:05d}.bin")
        out = self.tmp / "out_rebuilt"
        T4Converter4.convert(
            _config(self.t4, out, keyframes_only=True, lidar_format="point-cloud", rebuilt_lidar_dir=str(rebuilt))
        )
        pcs = _open(out).open_component_readers(PointCloudsComponent.Reader)["LIDAR_CONCAT"]
        key_ordinals = [i for i, sd in enumerate(lidar_sds) if sd["is_key_frame"]]
        self.assertEqual(pcs.pcs_count, len(key_ordinals))
        for pc_index, ordinal in enumerate(key_ordinals):
            np.testing.assert_array_equal(pcs.get_pc_xyz(pc_index)[:, 0], 1.0 + ordinal)
            ts = pcs.get_pc_attribute(pc_index, "timestamp_us")
            np.testing.assert_array_equal(ts - ts[0], [0, 10_000, 20_000, 30_000])


if __name__ == "__main__":
    unittest.main()
