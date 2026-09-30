# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import gaussian_factory per-frame derived artifacts into NCore camera labels.

gaussian_factory (3DGS reconstruction over T4 logs) writes several per-camera-frame
artifacts next to the raw T4 dataset. Storing them inside the NCore sequence lets the
raw T4 tree be dropped. Each artifact is described by a :class:`GfLayer`; files are
located by one of three key schemes used by gaussian_factory:

- ``stem``: ``<layer_dir>/<CAM>/<image_stem><suffix>`` (image filename stem)
- ``keyframe_index``: ``<layer_dir>/<CAM>/frame_{keyframe_index:04d}<suffix>``
  (ordinal of the ``sample`` in the scene's linked list)
- ``sd_token``: ``<layer_dir>/<sample_data_token><suffix>`` (no camera subdir)
"""

from __future__ import annotations

import io
import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np

from PIL import Image as PILImage

from ncore.impl.data.types import (
    CameraLabelDescriptor,
    JsonLike,
    LabelCategory,
    LabelEncoding,
    LabelSchema,
    LabelSource,
    LabelType,
    LabelUnit,
    QuantizationParams,
)


logger = logging.getLogger(__name__)

KeyScheme = Literal["stem", "keyframe_index", "sd_token"]
DepthEncoding = Literal["float32", "uint16"]

# uint16 depth quantization: 1/256 m (~3.9 mm) steps, 0..255.996 m range. Well below
# LiDAR range noise, and 0 keeps meaning "invalid".
DEPTH_UINT16_SCALE_M: float = 1.0 / 256.0


@dataclass(frozen=True)
class GfLayer:
    """One gaussian_factory artifact type and how it maps to an NCore camera label."""

    name: str  # CLI / instance-name identifier
    default_subdir: str  # relative to the gaussian_factory output root
    key: KeyScheme
    suffix: str
    kind: Literal["png", "depth"]
    label_type: LabelType
    description: str
    # side-car JSON files (relative to the layer dir) copied into the component meta-data
    sidecar_json: tuple[str, ...] = ()


GF_LAYERS: Dict[str, GfLayer] = {
    layer.name: layer
    for layer in (
        GfLayer(
            name="instance",
            default_subdir="sam_masks",
            key="stem",
            suffix=".png",
            kind="png",
            label_type=LabelType.SEGMENTATION_INSTANCE,
            description="SAM3 dynamic-object instance ids (uint8, 0=background, value=track id, "
            "see generic meta 'track_id_mapping' for instance_token -> id)",
            sidecar_json=("track_id_mapping.json",),
        ),
        GfLayer(
            name="sky",
            default_subdir="sky_masks",
            key="stem",
            suffix=".png",
            kind="png",
            label_type=LabelType(LabelCategory.MASK, "sky", LabelUnit.UNITLESS),
            description="SAM3 sky mask (uint8, 255=sky, 0=not sky)",
            sidecar_json=("_sky_text_prompts.json",),
        ),
        GfLayer(
            name="lidar_depth",
            default_subdir="lidar_depth_accum",
            key="keyframe_index",
            suffix=".npy",
            kind="depth",
            label_type=LabelType(LabelCategory.DEPTH, "z_lidar_accum", LabelUnit.METERS),
            description="Accumulated static LIDAR_CONCAT projected to camera z-depth [m], 0=invalid",
        ),
        GfLayer(
            name="mapanything_depth",
            default_subdir="mapanything_init_v14b/view_depth",
            key="sd_token",
            suffix=".npy",
            kind="depth",
            label_type=LabelType(LabelCategory.DEPTH, "z_mapanything", LabelUnit.METERS),
            description="MapAnything masked metric z-depth [m] at MapAnything resolution, 0=invalid",
        ),
    )
}


@dataclass(frozen=True)
class CameraFrameRef:
    """Identifies one stored camera frame for artifact lookup."""

    camera_id: str
    timestamp_us: int
    sample_data_token: str
    image_stem: str
    keyframe_index: Optional[int]


class GfLayerImporter:
    """Resolves and encodes gaussian_factory artifacts of a single layer."""

    def __init__(
        self,
        layer: GfLayer,
        layer_dir: Path,
        depth_encoding: DepthEncoding = "uint16",
        png_reoptimize: bool = True,
    ) -> None:
        self.layer = layer
        self.layer_dir = layer_dir
        self.depth_encoding = depth_encoding
        self.png_reoptimize = png_reoptimize
        self.n_depth_clipped = 0

    def path_for(self, frame: CameraFrameRef) -> Optional[Path]:
        if self.layer.key == "stem":
            return self.layer_dir / frame.camera_id / f"{frame.image_stem}{self.layer.suffix}"
        if self.layer.key == "keyframe_index":
            if frame.keyframe_index is None:
                return None
            return self.layer_dir / frame.camera_id / f"frame_{frame.keyframe_index:04d}{self.layer.suffix}"
        return self.layer_dir / f"{frame.sample_data_token}{self.layer.suffix}"

    def has_camera(self, camera_id: str) -> bool:
        if self.layer.key == "sd_token":
            return self.layer_dir.is_dir()
        return (self.layer_dir / camera_id).is_dir()

    def descriptor(self, camera_id: str) -> CameraLabelDescriptor:
        if self.layer.kind == "png":
            schema = LabelSchema(
                dtype=np.dtype("uint8"),
                encoding=LabelEncoding.IMAGE_ENCODED,
                encoded_format="png",
            )
        elif self.depth_encoding == "uint16":
            schema = LabelSchema(
                dtype=np.dtype("float32"),
                encoding=LabelEncoding.RAW,
                quantization=QuantizationParams(
                    quantized_dtype=np.dtype("uint16"),
                    scale=DEPTH_UINT16_SCALE_M,
                    offset=0.0,
                ),
            )
        else:
            schema = LabelSchema(dtype=np.dtype("float32"), encoding=LabelEncoding.RAW)

        return CameraLabelDescriptor(
            camera_id=camera_id,
            label_type=self.layer.label_type,
            label_schema=schema,
            label_source=LabelSource.AUTOLABEL,
        )

    def component_meta(self) -> Dict[str, JsonLike]:
        meta: Dict[str, JsonLike] = {
            "producer": "gaussian_factory",
            "gf_layer": self.layer.name,
            "gf_key_scheme": self.layer.key,
            "description": self.layer.description,
        }
        for name in self.layer.sidecar_json:
            if (p := self.layer_dir / name).is_file():
                with p.open("r") as f:
                    meta[Path(name).stem.lstrip("_")] = json.load(f)
        return meta

    def load(self, path: Path) -> "bytes | np.ndarray":
        if self.layer.kind == "png":
            data = path.read_bytes()
            if not self.png_reoptimize:
                return data
            # lossless re-encode, keep whichever is smaller
            with PILImage.open(io.BytesIO(data)) as im:
                if im.mode != "L":
                    im = im.convert("L")
                buf = io.BytesIO()
                im.save(buf, format="png", optimize=True)
            reencoded = buf.getvalue()
            return reencoded if len(reencoded) < len(data) else data

        depth = np.load(path).astype(np.float32, copy=False)
        if depth.ndim != 2:
            raise ValueError(f"{path}: expected (H, W) depth, got shape {depth.shape}")
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
        if self.depth_encoding == "uint16":
            max_m = np.iinfo(np.uint16).max * DEPTH_UINT16_SCALE_M
            too_far = depth > max_m
            if too_far.any():
                self.n_depth_clipped += int(too_far.sum())
                depth[too_far] = 0.0
        return depth


def resolve_layers(
    gf_root: Optional[Path],
    layer_names: Optional[List[str]],
    layer_dir_overrides: Dict[str, Path],
    depth_encoding: DepthEncoding,
    png_reoptimize: bool,
) -> List[GfLayerImporter]:
    """Build importers for requested layers (or every layer whose directory exists)."""
    unknown = (set(layer_names or []) | set(layer_dir_overrides)) - set(GF_LAYERS)
    if unknown:
        raise ValueError(f"Unknown gaussian_factory layer(s) {sorted(unknown)}; known: {sorted(GF_LAYERS)}")

    importers: List[GfLayerImporter] = []
    for name, layer in GF_LAYERS.items():
        if layer_names and name not in layer_names:
            continue
        if name in layer_dir_overrides:
            layer_dir = layer_dir_overrides[name]
        elif gf_root is not None:
            layer_dir = gf_root / layer.default_subdir
        else:
            continue
        if not layer_dir.is_dir():
            if layer_names and name in layer_names:
                raise FileNotFoundError(f"gaussian_factory layer '{name}' directory not found: {layer_dir}")
            logger.info(f"gaussian_factory layer '{name}' not found at {layer_dir}, skipping")
            continue
        importers.append(GfLayerImporter(layer, layer_dir, depth_encoding, png_reoptimize))
    return importers


def load_camera_poses_json(path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, JsonLike]]:
    """Load a gaussian_factory ``{sample_data_token: 4x4 camera->world}`` pose file.

    Accepts both the flat layout (``trajectory_correction/poses.json``) and the trainer's
    ``refined_poses_step*.json`` layout (``{"meta": ..., "poses": {...}}``). Returns
    ``(poses, meta)``.
    """
    with path.open("r") as f:
        root = json.load(f)
    nested = isinstance(root, dict) and isinstance(root.get("poses"), dict)
    poses = root["poses"] if nested else root
    meta = root.get("meta", {}) if nested else {}
    out: Dict[str, np.ndarray] = {}
    for token, mat in poses.items():
        arr = np.asarray(mat, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError(f"{path}: pose for {token} has shape {arr.shape}, expected (4, 4)")
        out[token] = arr
    return out, meta
