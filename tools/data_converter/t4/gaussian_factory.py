# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gaussian_factory preprocessing artifacts and how they map onto NCore components.

Only what ``train_static_near_streaming`` consumes is covered (masks / depth as camera
labels, the MapAnything initial Gaussians as a point cloud). Each artifact is a
:class:`GfLayer`; per-frame files are located by one of three key schemes used by
gaussian_factory:

- ``stem``: ``<layer_dir>/<CAM>/<image_stem><suffix>`` (image filename stem)
- ``keyframe_index``: ``<layer_dir>/<CAM>/frame_{keyframe_index:04d}<suffix>``
  (ordinal of the ``sample`` in the scene's linked list)
- ``sd_token``: ``<layer_dir>/<sample_data_token><suffix>`` (no camera subdir)

The frame identity (timestamp, sample_data token, image stem, keyframe index) comes from
the camera frames' generic meta-data of an already converted NCore sequence, so layers can
be appended without the raw T4 tree.
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np

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
from tools.data_converter.t4.utils import reencode_if_smaller


logger = logging.getLogger(__name__)

KeyScheme = Literal["stem", "keyframe_index", "sd_token", "none"]
DepthEncoding = Literal["float32", "uint16"]

# uint16 depth quantization: 1/256 m (~3.9 mm) steps, 0..255.996 m range. Well below
# LiDAR range noise, and 0 keeps meaning "invalid".
DEPTH_UINT16_SCALE_M: float = 1.0 / 256.0


@dataclass(frozen=True)
class GfLayer:
    """One gaussian_factory artifact type and how it maps to an NCore component."""

    name: str  # CLI identifier, also names the store group ``gf_<name>``
    default_path: str  # relative to the gaussian_factory output root (dir, or file for "gaussians")
    key: KeyScheme
    suffix: str
    kind: Literal["png", "depth", "gaussians"]
    description: str
    label_type: Optional[LabelType] = None  # camera-label layers only
    # side-car JSON files (relative to the layer dir) copied into the component meta-data
    sidecar_json: tuple[str, ...] = ()


GF_LAYERS: Dict[str, GfLayer] = {
    layer.name: layer
    for layer in (
        GfLayer(
            name="instance",
            default_path="sam_masks",
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
            default_path="sky_masks",
            key="stem",
            suffix=".png",
            kind="png",
            label_type=LabelType(LabelCategory.MASK, "sky", LabelUnit.UNITLESS),
            description="SAM3 sky mask (uint8, 255=sky, 0=not sky)",
            sidecar_json=("_sky_text_prompts.json",),
        ),
        GfLayer(
            name="lidar_depth",
            default_path="lidar_depth_accum",
            key="keyframe_index",
            suffix=".npy",
            kind="depth",
            label_type=LabelType(LabelCategory.DEPTH, "z_lidar_accum", LabelUnit.METERS),
            description="Accumulated static LIDAR_CONCAT projected to camera z-depth [m], 0=invalid",
        ),
        GfLayer(
            name="mapanything_depth",
            default_path="mapanything_init_v14b/view_depth_masked",
            key="sd_token",
            suffix=".npy",
            kind="depth",
            label_type=LabelType(LabelCategory.DEPTH, "z_mapanything", LabelUnit.METERS),
            description="MapAnything masked metric z-depth [m] at MapAnything resolution, 0=invalid",
        ),
        GfLayer(
            name="init_gaussians",
            default_path="mapanything_init_v14b/initial_gaussians.ply",
            key="none",
            suffix=".ply",
            kind="gaussians",
            description="MapAnything initial 3DGS (T4 world frame); every non-xyz PLY vertex property "
            "is a float attribute of the same name",
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
        path: Path,
        depth_encoding: DepthEncoding = "uint16",
        png_reoptimize: bool = True,
    ) -> None:
        self.layer = layer
        self.path = path  # layer directory, or the PLY file for "gaussians"
        self.depth_encoding = depth_encoding
        self.png_reoptimize = png_reoptimize

    @property
    def is_camera_label(self) -> bool:
        return self.layer.kind in ("png", "depth")

    def path_for(self, frame: CameraFrameRef) -> Optional[Path]:
        if self.layer.key == "stem":
            return self.path / frame.camera_id / f"{frame.image_stem}{self.layer.suffix}"
        if self.layer.key == "keyframe_index":
            if frame.keyframe_index is None:
                return None
            return self.path / frame.camera_id / f"frame_{frame.keyframe_index:04d}{self.layer.suffix}"
        return self.path / f"{frame.sample_data_token}{self.layer.suffix}"

    def descriptor(self, camera_id: str) -> CameraLabelDescriptor:
        assert self.layer.label_type is not None
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
                    # exact: depth * 256 is a power-of-two scaling and 65535 fits float32's mantissa
                    intermediate_dtype=np.dtype("float32"),
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
            "gf_source": str(self.path),
            "description": self.layer.description,
        }
        if self.layer.kind == "depth":
            meta["gf_depth_encoding"] = self.depth_encoding
        for name in self.layer.sidecar_json:
            if (p := self.path / name).is_file():
                with p.open("r") as f:
                    meta[Path(name).stem.lstrip("_")] = json.load(f)
        return meta

    def load(self, path: Path) -> "tuple[bytes | np.ndarray, int]":
        """Encoded label for ``path`` and the number of depth pixels beyond the uint16 range set invalid."""
        if self.layer.kind == "png":
            data = path.read_bytes()
            # lossless re-encode, keep whichever is smaller
            return (reencode_if_smaller(data, "png", "L", optimize=True) if self.png_reoptimize else data), 0

        depth = np.load(path).astype(np.float32, copy=False)
        if depth.ndim != 2:
            raise ValueError(f"{path}: expected (H, W) depth, got shape {depth.shape}")
        if not depth.flags.writeable:
            depth = depth.copy()
        invalid = ~(depth > 0)  # also catches NaN
        invalid |= np.isinf(depth)
        depth[invalid] = 0.0
        n_clipped = 0
        if self.depth_encoding == "uint16":
            too_far = depth > np.iinfo(np.uint16).max * DEPTH_UINT16_SCALE_M
            n_clipped = int(np.count_nonzero(too_far))
            if n_clipped:
                depth[too_far] = 0.0
        return depth, n_clipped


def resolve_layers(
    gf_root: Optional[Path],
    layer_names: Optional[List[str]],
    layer_path_overrides: Dict[str, Path],
    depth_encoding: DepthEncoding,
    png_reoptimize: bool,
) -> List[GfLayerImporter]:
    """Build importers for requested layers (or every layer whose source exists)."""
    unknown = (set(layer_names or []) | set(layer_path_overrides)) - set(GF_LAYERS)
    if unknown:
        raise ValueError(f"Unknown gaussian_factory layer(s) {sorted(unknown)}; known: {sorted(GF_LAYERS)}")

    importers: List[GfLayerImporter] = []
    for name, layer in GF_LAYERS.items():
        if layer_names and name not in layer_names:
            continue
        if name in layer_path_overrides:
            path = layer_path_overrides[name]
        elif gf_root is not None:
            path = gf_root / layer.default_path
        else:
            continue
        exists = path.is_file() if layer.kind == "gaussians" else path.is_dir()
        if not exists:
            if layer_names:
                raise FileNotFoundError(f"gaussian_factory layer '{name}' not found: {path}")
            logger.info(f"gaussian_factory layer '{name}' not found at {path}, skipping")
            continue
        importers.append(GfLayerImporter(layer, path, depth_encoding, png_reoptimize))
    return importers


_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def read_ply_vertices(path: Path) -> np.ndarray:
    """Read the ``vertex`` element of a binary little-endian PLY into a structured array."""
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt = None
        elements: List[tuple[str, int, List[tuple[str, str]]]] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            tokens = line.decode("ascii").split()
            if not tokens or tokens[0] in ("comment", "obj_info"):
                continue
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                elements.append((tokens[1], int(tokens[2]), []))
            elif tokens[0] == "property":
                if tokens[1] == "list":
                    raise ValueError(f"{path}: list properties are not supported")
                elements[-1][2].append((tokens[2], "<" + _PLY_TYPES[tokens[1]]))
            elif tokens[0] == "end_header":
                break
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: only binary_little_endian PLY is supported, got {fmt}")
        for name, count, props in elements:
            data = np.fromfile(f, dtype=np.dtype(props), count=count)
            if name == "vertex":
                return data
    raise ValueError(f"{path}: no vertex element")
