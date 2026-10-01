# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append gaussian_factory preprocessing layers to an existing T4 NCore sequence.

NCore V4 component stores are write-once, but a sequence is a *set* of stores that share
``sequence_id``, time interval and sequence meta-data. Each layer is therefore written as
its own store ``<sequence_id>.ncore4-gf_<layer>.zarr.itar`` via
:meth:`SequenceComponentGroupsWriter.from_reader`; re-appending a layer replaces that
store. The sequence meta JSON (``<sequence_id>.json``, the store manifest) is rewritten
afterwards, so opening the JSON always yields the base conversion plus all layers.

    python -m tools.data_converter.t4.gf_append \
        --sequence-dir <ncore_out>/<sequence_id> --gf-root <gaussian_factory scene output> \
        [--layer sky --layer instance ...] [--layer-path lidar_depth=/run/lidar_depth_accum]
"""

from __future__ import annotations

import json
import logging
import shutil

from pathlib import Path
from typing import Dict, List, Literal, Optional

import click
import numpy as np
import tqdm

from upath import UPath

from ncore.impl.data.types import JsonLike, PointCloud
from ncore.impl.data.v4.components import (
    CameraLabelsComponent,
    CameraSensorComponent,
    PointCloudsComponent,
    SequenceComponentGroupsReader,
    SequenceComponentGroupsWriter,
    SequenceMeta,
)
from tools.data_converter.t4.gaussian_factory import (
    GF_LAYERS,
    CameraFrameRef,
    DepthEncoding,
    GfLayerImporter,
    read_ply_vertices,
    resolve_layers,
)


logger = logging.getLogger(__name__)


def find_sequence_meta(sequence_dir: Path) -> Path:
    """Locate the ``<sequence_id>.json`` store manifest of a converted sequence."""
    candidates = []
    for p in sorted(sequence_dir.glob("*.json")):
        try:
            with p.open("r") as f:
                root = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(root, dict) and "component_stores" in root and "sequence_id" in root:
            candidates.append(p)
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one sequence meta JSON in {sequence_dir}, found {candidates}")
    return candidates[0]


def camera_frames_from_reader(reader: SequenceComponentGroupsReader) -> Dict[str, List[CameraFrameRef]]:
    """Rebuild per-camera frame identities from the T4 provenance stored by the converter."""
    frames: Dict[str, List[CameraFrameRef]] = {}
    for camera_id, cam in reader.open_component_readers(CameraSensorComponent.Reader).items():
        refs = []
        for ts in cam.frames_timestamps_us[:, 1]:
            meta = cam.get_frame_generic_meta_data(int(ts))
            if "t4_sample_data_token" not in meta:
                raise ValueError(f"{camera_id}: frame {ts} has no T4 provenance meta-data (not a T4 conversion?)")
            kf = meta.get("t4_keyframe_index")
            refs.append(
                CameraFrameRef(
                    camera_id=camera_id,
                    timestamp_us=int(ts),
                    sample_data_token=str(meta["t4_sample_data_token"]),
                    image_stem=Path(str(meta["t4_filename"])).stem,
                    keyframe_index=int(kf) if isinstance(kf, (int, float)) else None,
                )
            )
        frames[camera_id] = refs
    return frames


def _write_camera_labels(
    importer: GfLayerImporter,
    writer: SequenceComponentGroupsWriter,
    group_name: str,
    camera_frames: Dict[str, List[CameraFrameRef]],
) -> int:
    component_meta = importer.component_meta()
    n_total = 0
    for camera_id, frames in sorted(camera_frames.items()):
        if not importer.has_camera(camera_id):
            continue
        located = [(f, p) for f in frames if (p := importer.path_for(f)) is not None and p.is_file()]
        if not located:
            continue
        descriptor = importer.descriptor(camera_id)
        label_writer = writer.register_component_writer(
            CameraLabelsComponent.Writer,
            component_instance_name=descriptor.default_instance_name,
            group_name=group_name,
            generic_meta_data=component_meta,
            descriptor=descriptor,
        )
        for frame, path in tqdm.tqdm(located, desc=f"gf {importer.layer.name} {camera_id}"):
            label_writer.store_label(
                importer.load(path),
                timestamp_us=frame.timestamp_us,
                generic_meta_data={"t4_sample_data_token": frame.sample_data_token},
            )
        logger.info(
            f"gaussian_factory {importer.layer.name}@{camera_id}: stored {len(located)}, "
            f"no artifact for {len(frames) - len(located)} frames"
        )
        n_total += len(located)
    if importer.n_depth_clipped:
        logger.warning(
            f"gaussian_factory {importer.layer.name}: {importer.n_depth_clipped} depth px beyond uint16 range "
            "set invalid"
        )
    return n_total


def _write_gaussians(
    importer: GfLayerImporter,
    writer: SequenceComponentGroupsWriter,
    group_name: str,
    reference_timestamp_us: int,
) -> int:
    vertices = read_ply_vertices(importer.path)
    names = list(vertices.dtype.names or ())
    if not {"x", "y", "z"} <= set(names):
        raise ValueError(f"{importer.path}: PLY vertex element has no x/y/z")
    attributes = {n: np.ascontiguousarray(vertices[n]) for n in names if n not in ("x", "y", "z")}
    component_meta = importer.component_meta()
    component_meta["ply_properties"] = list[JsonLike](names)
    pc_writer = writer.register_component_writer(
        PointCloudsComponent.Writer,
        component_instance_name=f"gf_{importer.layer.name}",
        group_name=group_name,
        generic_meta_data=component_meta,
        coordinate_unit=PointCloud.CoordinateUnit.METERS,
        attribute_schemas={
            n: PointCloudsComponent.AttributeSchema(
                transform_type=PointCloud.AttributeTransformType.INVARIANT, dtype=a.dtype
            )
            for n, a in attributes.items()
        },
    )
    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    pc_writer.store_pc(
        xyz=xyz,
        reference_frame_id="world",
        reference_frame_timestamp_us=reference_timestamp_us,
        attributes=attributes,
    )
    logger.info(f"gaussian_factory {importer.layer.name}: stored {len(xyz):,} Gaussians")
    return len(xyz)


def _remove_store(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def append_gf_layers(
    sequence_dir: Path,
    gf_root: Optional[Path],
    layer_names: Optional[List[str]] = None,
    layer_paths: Optional[Dict[str, Path]] = None,
    depth_encoding: DepthEncoding = "uint16",
    png_reoptimize: bool = True,
    store_type: Literal["itar", "directory"] = "itar",
) -> Dict[str, int]:
    """Append (or replace) gaussian_factory layers as extra stores; returns items stored per layer."""
    meta_path = find_sequence_meta(sequence_dir)
    with meta_path.open("r") as f:
        sequence_meta = SequenceMeta.from_dict(json.load(f))
    store_paths = [sequence_dir / s.path for s in sequence_meta.component_stores]

    reader = SequenceComponentGroupsReader([UPath(p) for p in store_paths])
    importers = resolve_layers(gf_root, layer_names, layer_paths or {}, depth_encoding, png_reoptimize)
    if not importers:
        raise FileNotFoundError(f"no gaussian_factory layers found (gf_root={gf_root}, layers={layer_names})")
    camera_frames = camera_frames_from_reader(reader) if any(i.is_camera_label for i in importers) else {}

    stored: Dict[str, int] = {}
    for importer in importers:
        name = importer.layer.name
        group_name = f"gf_{name}"
        tmp_dir = sequence_dir / f".gf_append_{name}.tmp"
        _remove_store(tmp_dir)
        writer = SequenceComponentGroupsWriter.from_reader(
            output_dir_path=UPath(tmp_dir),
            store_base_name=reader.sequence_id,
            sequence_reader=reader,
            store_type=store_type,
        )
        if importer.is_camera_label:
            stored[name] = _write_camera_labels(importer, writer, group_name, camera_frames)
        else:
            stored[name] = _write_gaussians(importer, writer, group_name, reader.sequence_timestamp_interval_us.start)
        if stored[name] == 0:
            logger.warning(f"gaussian_factory {name}: nothing matched the sequence frames, layer not written")
            _remove_store(tmp_dir)
            continue
        for new_path in writer.finalize():
            target = sequence_dir / new_path.name
            if target.exists():
                logger.info(f"replacing existing store {target.name}")
            _remove_store(target)
            shutil.move(str(new_path), str(target))
            if target not in store_paths:
                store_paths.append(target)
        _remove_store(tmp_dir)

    # rewrite the store manifest (atomically) so the JSON lists base + all layers
    updated = SequenceComponentGroupsReader([UPath(p) for p in store_paths]).get_sequence_meta()
    tmp_meta = meta_path.with_suffix(".json.tmp")
    with tmp_meta.open("w") as f:
        json.dump(updated.to_dict(), f, indent=2)
    tmp_meta.replace(meta_path)
    return stored


@click.command()
@click.option(
    "--sequence-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Converted NCore sequence directory (holding <sequence_id>.json).",
)
@click.option(
    "--gf-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="gaussian_factory output root of the scene; every layer found there is appended.",
)
@click.option("--layer", "layers", multiple=True, type=click.Choice(sorted(GF_LAYERS)), help="Layers to append.")
@click.option(
    "--layer-path",
    "layer_paths",
    multiple=True,
    help="Override a layer source as NAME=PATH (directory, or the PLY file for init_gaussians).",
)
@click.option(
    "--depth-encoding",
    type=click.Choice(["uint16", "float32"]),
    default="uint16",
    show_default=True,
    help="uint16 quantizes depth to 1/256 m (range 256 m); float32 is lossless.",
)
@click.option("--png-reoptimize/--no-png-reoptimize", default=True, show_default=True)
@click.option("--store-type", type=click.Choice(["itar", "directory"]), default="itar", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def main(
    sequence_dir: Path,
    gf_root: Optional[Path],
    layers: tuple[str, ...],
    layer_paths: tuple[str, ...],
    depth_encoding: DepthEncoding,
    png_reoptimize: bool,
    store_type: Literal["itar", "directory"],
    verbose: bool,
) -> None:
    """Append gaussian_factory layers to a converted T4 NCore sequence."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    overrides: Dict[str, Path] = {}
    for spec in layer_paths:
        name, sep, path = spec.partition("=")
        if not sep:
            raise click.BadParameter(f"expected NAME=PATH, got {spec!r}", param_hint="--layer-path")
        overrides[name] = Path(path)
    stored = append_gf_layers(
        sequence_dir, gf_root, list(layers), overrides, depth_encoding, png_reoptimize, store_type
    )
    for name, n in stored.items():
        click.echo(f"{name}: {n}")


if __name__ == "__main__":
    main()
