<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# T4 Dataset Converter

Convert a TIER IV T4 dataset into NCore V4 component stores, and append the
preprocessing artifacts of
[gaussian_factory](https://github.com/hakuturu583/gaussian_factory) that its streaming
trainer consumes (masks, depth, initial Gaussians), so that the raw T4 tree can be
replaced by a smaller NCore sequence.

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

### gaussian_factory layers (appended afterwards)

NCore V4 stores are write-once, but a sequence is a *set* of stores sharing
`sequence_id`, time interval and sequence meta-data. The base conversion therefore keeps
the sequence meta-data invariant (T4 identifiers only; conversion options live in the
component meta-data), and each gaussian_factory layer is appended later as its own
store `<sequence_id>.ncore4-gf_<layer>.zarr.itar`, e.g. right after the pipeline step that
produced it:

```bash
python -m tools.data_converter.t4.gf_append --sequence-dir $OUT/<sequence_id> \
    --gf-root $GF_ROOT [--layer sky --layer instance] [--layer-path NAME=PATH]
```

Re-appending a layer replaces its store. The sequence meta JSON `<sequence_id>.json` is
rewritten as the store manifest, so opening it with `SequenceComponentGroupsReader`
yields the base conversion plus every appended layer. Appending needs only the NCore
sequence and the gaussian_factory output (frames are matched via the T4 provenance in the
camera frame meta-data), not the raw T4 tree.

Only what `train_static_near_streaming` reads is covered:

| layer | source (default under `--gf-root`) | key | NCore |
|---|---|---|---|
| `instance` | `sam_masks/<CAM>/<stem>.png` (+`track_id_mapping.json`) | image stem | label `segmentation.instance@<CAM>`, PNG uint8 |
| `sky` | `sky_masks/<CAM>/<stem>.png` | image stem | label `mask.sky@<CAM>`, PNG uint8 (255=sky) |
| `lidar_depth` | `lidar_depth_accum/<CAM>/frame_NNNN.npy` | keyframe ordinal | label `depth.z_lidar_accum@<CAM>` |
| `mapanything_depth` | `mapanything_init_v14b/view_depth_masked/<sd_token>.npy` | sample_data token | label `depth.z_mapanything@<CAM>` |
| `init_gaussians` | `mapanything_init_v14b/initial_gaussians.ply` | - | point cloud `gf_init_gaussians` (world frame, every PLY property as an attribute) |

- Depth is stored as uint16 in 1/256 m steps (range 256 m, 0 = invalid) by default;
  `--depth-encoding float32` stores it losslessly.
- PNGs are losslessly re-optimized (`--no-png-reoptimize` keeps source bytes).
- Labels use the camera frame timestamp and carry their `t4_sample_data_token`.
- Not covered (not read by the streaming trainer, or rebuilt by it): static mesh,
  quadmask / inpainting, dynamic_objects_v2, near masks, ground-cloud / visibility-voxel /
  VAD caches, cuVSLAM / refined poses.

Typical gaussian_factory flow:

```bash
python -m tools.data_converter.t4.main --root-dir $T4 --output-dir $OUT \
    --camera-id CAM_FRONT --camera-id CAM_FRONT_LEFT --camera-id CAM_FRONT_RIGHT \
    --camera-id CAM_BACK_LEFT --camera-id CAM_BACK_RIGHT \
    t4-v4 --lidar-format point-cloud
python -m tools.data_converter.t4.gf_append --sequence-dir $OUT/<sequence_id> --gf-root $GF_ROOT
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
