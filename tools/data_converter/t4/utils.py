# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for parsing T4 dataset annotation JSON files."""

from __future__ import annotations

import io
import json

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from PIL import Image as PILImage
from upath import UPath

from ncore.impl.data.v4.components import SequenceComponentGroupsReader


_REQUIRED_FILES: tuple[str, ...] = (
    "scene",
    "sensor",
    "calibrated_sensor",
    "ego_pose",
    "sample",
    "sample_data",
)
_OPTIONAL_FILES: tuple[str, ...] = (
    "sample_annotation",
    "instance",
    "category",
    "visibility",
)
ANNOTATION_FILES: tuple[str, ...] = _REQUIRED_FILES + _OPTIONAL_FILES


def load_annotation_tables(annotation_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Read T4 annotation JSON files under ``annotation_dir`` into a dict of lists.

    Required files raise on missing. Optional annotation files return an empty
    list when absent.
    """
    tables: Dict[str, List[Dict[str, Any]]] = {}
    for name in _REQUIRED_FILES:
        with (annotation_dir / f"{name}.json").open("r") as f:
            tables[name] = json.load(f)
    for name in _OPTIONAL_FILES:
        path = annotation_dir / f"{name}.json"
        tables[name] = json.load(path.open("r")) if path.exists() else []
    return tables


def index_by_token(table: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index a T4 table by its primary ``token`` field."""
    return {row["token"]: row for row in table}


def quaternion_wxyz_to_so3(q: np.ndarray) -> np.ndarray:
    """Convert a ``[w, x, y, z]`` quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm == 0.0:
        raise ValueError("Zero-norm quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def t4_pose_to_se3(translation: List[float], rotation_wxyz: List[float]) -> np.ndarray:
    """Build a 4x4 SE(3) source->target matrix from a (translation, quaternion[wxyz]) pair."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_wxyz_to_so3(np.asarray(rotation_wxyz, dtype=np.float64))
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


_T4_LIDAR_FLOATS_PER_POINT = 5


def load_lidar_points(path: Path) -> np.ndarray:
    """Load a ``LIDAR_CONCAT/*.pcd.bin`` file as an Nx5 float32 (x, y, z, intensity, ring) array."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % _T4_LIDAR_FLOATS_PER_POINT != 0:
        raise ValueError(
            f"Lidar point buffer size {raw.size} (path={path}) is not a multiple of {_T4_LIDAR_FLOATS_PER_POINT}"
        )
    return raw.reshape(-1, _T4_LIDAR_FLOATS_PER_POINT)


def keyframe_index_by_sample_token(samples: List[Dict[str, Any]], scene: Dict[str, Any]) -> Dict[str, int]:
    """Ordinal of each ``sample`` along the scene's ``first_sample_token`` -> ``next`` chain.

    gaussian_factory names keyframe artifacts ``frame_{ordinal:04d}``.
    """
    by_token = index_by_token(samples)
    out: Dict[str, int] = {}
    token = scene.get("first_sample_token", "")
    while token and token in by_token and token not in out:
        out[token] = len(out)
        token = by_token[token].get("next", "")
    return out


def t4_distortion_to_opencv(distortion: List[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a T4 ``camera_distortion`` vector into NCore OpenCV pinhole coefficients.

    T4 stores OpenCV-ordered coefficients ``[k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]``
    truncated to 0/4/5/8/12 entries. Returns ``(radial[6], tangential[2], thin_prism[4])``.
    """
    d = np.zeros(12, dtype=np.float32)
    n = min(len(distortion), 12)
    d[:n] = np.asarray(distortion[:n], dtype=np.float32)
    radial = np.array([d[0], d[1], d[4], d[5], d[6], d[7]], dtype=np.float32)
    tangential = np.array([d[2], d[3]], dtype=np.float32)
    thin_prism = d[8:12].copy()
    return radial, tangential, thin_prism


def reencode_if_smaller(data: bytes, format: str, mode: str, **save_kwargs: Any) -> bytes:
    """Re-encode an image as ``format`` (after converting to ``mode``); keep whichever is smaller."""
    with PILImage.open(io.BytesIO(data)) as im:
        buf = io.BytesIO()
        (im if im.mode == mode else im.convert(mode)).save(buf, format=format, **save_kwargs)
    reencoded = buf.getvalue()
    return reencoded if len(reencoded) < len(data) else data


def write_sequence_manifest(store_paths: Sequence[UPath | Path], meta_path: UPath | Path) -> None:
    """(Re)write the ``<sequence_id>.json`` store manifest listing ``store_paths``, atomically."""
    meta = SequenceComponentGroupsReader([UPath(p) for p in store_paths]).get_sequence_meta()
    meta_path = UPath(meta_path)
    tmp = meta_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(meta.to_dict(), f, indent=2)
    tmp.replace(meta_path)
