<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# T4 Dataset Converter

Convert a TIER IV T4 dataset into NCore V4 component stores, optionally together
with the per-frame artifacts produced by
[gaussian_factory](https://github.com/hakuturu583/gaussian_factory) (masks, depth,
refined poses), so that the raw T4 tree can be replaced by a smaller NCore sequence.

## Frame mapping

| T4 | NCore V4 |
|---|---|
| `base_link` | `rig` |
| `map` | `world` |
| sensor channel | sensor id (same string) |

`calibrated_sensor.{translation, rotation}` is stored as a static pose
`sensor -> rig`. `ego_pose` is stored as a dynamic pose `rig -> world`.
`camera_distortion` (OpenCV order `k1,k2,p1,p2,k3[,k4,k5,k6[,s1..s4]]`) is mapped
onto the `OpenCVPinholeCameraModelParameters` radial / tangential / thin-prism
coefficients.

Every camera / lidar frame carries T4 provenance in its generic meta-data
(`t4_sample_data_token`, `t4_sample_token`, `t4_filename`, `t4_is_key_frame`,
`t4_keyframe_index`). Cuboids keep `num_lidar_pts`, `num_radar_pts` and the
visibility level as component generic data aligned with the observation order.
`map/lanelet2_map.osm` is embedded as `t4_lanelet2_map_osm` generic data of the
default poses component.

## Usage

```bash
python -m tools.data_converter.t4.main \
    --root-dir /path/to/t4_sequence_or_parent_dir \
    --output-dir /path/to/ncore_output \
    t4-v4 [--label-source autolabel|gt-annotation|external]
```

`--root-dir` may point either at a single T4 sequence (a directory containing
`annotation/` and `data/`) or at a parent directory containing multiple such
sequences side-by-side. Sensors can be subselected with the common
`--camera-id` / `--lidar-id` / `--no-radars` options (before `t4-v4`).

### Storage reduction options

| Option | Effect |
|---|---|
| `--keyframes-only` | only `is_key_frame` sample_data (gaussian_factory uses keyframes only) |
| `--camera-id ...` | only the listed cameras (gaussian_factory uses 5, not `CAM_BACK`) |
| `--lidar-format point-cloud` | `PointCloudsComponent` with float32 xyz + uint8 intensity + int16 ring instead of the ray bundle (≈0.53× vs ≈0.74× of `.pcd.bin`) |
| `--jpeg-quality Q` | lossy JPEG re-encode (source bytes are kept by default) |

The rosbag (`input_bag/`) is never converted.

### gaussian_factory artifacts

`--gf-root <gaussian_factory output root of the scene>` imports every layer found
there as `CameraLabelsComponent` instances (one store per layer, so a layer can be
dropped or shipped separately):

| layer | source (default subdir) | key | NCore label |
|---|---|---|---|
| `instance` | `sam_masks/<CAM>/<stem>.png` (+`track_id_mapping.json`) | image stem | `segmentation.instance@<CAM>`, PNG uint8 |
| `sky` | `sky_masks/<CAM>/<stem>.png` | image stem | `mask.sky@<CAM>`, PNG uint8 (255=sky) |
| `lidar_depth` | `lidar_depth_accum/<CAM>/frame_NNNN.npy` | keyframe ordinal | `depth.z_lidar_accum@<CAM>` |
| `mapanything_depth` | `mapanything_init_v14b/view_depth/<sd_token>.npy` | sample_data token | `depth.z_mapanything@<CAM>` |

- `--gf-layer NAME` restricts the import, `--gf-layer-dir NAME=PATH` overrides a directory.
- Depth is stored as uint16 in 1/256 m steps (range 256 m, 0 = invalid) by default;
  `--gf-depth-encoding float32` stores it losslessly.
- PNGs are losslessly re-optimized (`--no-gf-png-reoptimize` keeps source bytes).
- `--gf-camera-poses-json` (`trajectory_correction/poses.json` or the trainer's
  `refined_poses_step*.json`, `{sample_data_token: camera->world}`) is stored as
  dynamic `<CAM> -> world` poses of a separate `gaussian_factory` poses component.

Labels use the camera frame timestamp; each label also carries its
`t4_sample_data_token`.

Typical gaussian_factory export:

```bash
python -m tools.data_converter.t4.main --root-dir $T4 --output-dir $OUT \
    --camera-id CAM_FRONT --camera-id CAM_FRONT_LEFT --camera-id CAM_FRONT_RIGHT \
    --camera-id CAM_BACK_LEFT --camera-id CAM_BACK_RIGHT \
    t4-v4 --keyframes-only --lidar-format point-cloud --gf-root $GF_ROOT
```

### Storage report

```bash
python -m tools.data_converter.t4.storage_report --t4 $T4 --gf-root $GF_ROOT \
    --ncore $OUT/<sequence> --probe-jpeg-quality 90 --probe-jpeg-quality 85
```

prints the T4 breakdown (rosbag, keyframe / non-keyframe data per channel, what
gaussian_factory needs), the gaussian_factory artifact sizes, the NCore store sizes,
and the JPEG re-encode ratio measured on sampled images.

### Label source

`sample_annotation.json` may contain either online detector outputs or
human-labeled ground truth depending on how the source dataset was produced.
Set `--label-source` to match (`autolabel` is the default).

### Real lidar ring + per-point time (optional)

By default the lidar comes from the T4 `LIDAR_CONCAT/*.pcd.bin`, which only has
`x, y, z, intensity, ring` — no per-point time. To recover it, decode the raw
`pandar_packets` first (see [`lidar_rebuild/`](lidar_rebuild/)) and pass the result:

```bash
... t4-v4 --rebuilt-lidar-dir <lidar_rebuild>/concatenated
```

The converter then stores real per-point timestamps. Do this before discarding
`input_bag/`, which is the only source of the raw packets.

## Limitations

- Camera frame intervals are stored as instantaneous (`[ts, ts]`); shutter
  readout duration is not represented.
- Without `--rebuilt-lidar-dir`, per-ray lidar timestamps default to the
  scan-start timestamp (the T4 `.pcd.bin` has no per-point time).
- The fused `LIDAR_CONCAT` cloud is stored unstructured (no model element); the
  T4 ring index is kept as per-frame generic data (`ring`) / point attribute.
- Cuboid annotations follow the nuScenes-style schema. Image-space annotations
  (`object_ann`, `surface_ann`) are not converted.
