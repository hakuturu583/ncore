# NCore T4 Converter

Convert a T4 dataset loaded through `t4-devkit` 0.7.0 into NCore V4 format.

## Usage

```bash
bazel run //tools/data_converter/t4:convert -- \
    --root-dir /path/to/t4-dataset \
    --output-dir /path/to/ncore-output \
    t4-v4
```

To target a specific T4 dataset revision or scene:

```bash
bazel run //tools/data_converter/t4:convert -- \
    --root-dir /path/to/t4-dataset \
    --output-dir /path/to/ncore-output \
    t4-v4 \
    --revision 2 \
    --scene-name my_scene
```

To also export a PCD map as a `PointCloudsComponent`, provide a PCD file or
directory:

```bash
bazel run //tools/data_converter/t4:convert -- \
    --root-dir /path/to/t4-dataset \
    --output-dir /path/to/ncore-output \
    --pcd-map-path /path/to/map.pcd \
    t4-v4
```

If `--pcd-map-path` is omitted, the converter tries to auto-discover a unique
`.pcd` file under the T4 root, including the conventional
`annotation_dataset/<dataset_id>/<version>/map` directory. For multi-scene
datasets, passing the map path explicitly is still safer.

## Conversion Mapping

- `scene` -> one NCore sequence
- `ego_pose` -> `PosesComponent` dynamic `rig -> world`
- `calibrated_sensor` + image `sample_data` -> camera components and intrinsics
- LiDAR / Radar `sample_data` -> `PointCloudsComponent` per sensor
- `sample_annotation` -> `CuboidsComponent`

## Notes

- The converter assumes T4 camera distortion coefficients follow the OpenCV pinhole ordering used by `camera_distortion`.
- T4 LiDAR point clouds are stored as native point clouds in NCore rather than ray bundles because the source format is already point-based.
