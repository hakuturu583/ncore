# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage breakdown of a T4 sequence vs. its NCore conversion.

Reports which parts of the raw T4 tree a gaussian_factory-style consumer needs
(keyframes of the selected cameras + LIDAR_CONCAT + annotation), the size of the
gaussian_factory artifacts, and — if given — the size of each NCore store.

    python -m tools.data_converter.t4.storage_report \
        --t4 <t4_sequence> [--gf-root <gf_output>] [--ncore <ncore_sequence_dir>] \
        [--camera-id CAM_FRONT ...] [--probe-jpeg-quality 90 --probe-jpeg-quality 85]
"""

from __future__ import annotations

import io
import json

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import click
import numpy as np

from PIL import Image as PILImage

from tools.data_converter.t4.gaussian_factory import GF_LAYERS
from tools.data_converter.t4.utils import index_by_token, load_annotation_tables


def _du(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:8.1f} {unit}"
        n /= 1024
    return ""


def _row(label: str, size: float, total: float) -> str:
    return f"  {label:<46s}{_fmt(size)}  {100 * size / total:5.1f}%" if total else f"  {label:<46s}{_fmt(size)}"


def _probe_jpeg(paths: List[Path], qualities: Iterable[int]) -> Dict[int, float]:
    """Mean re-encoded/original size ratio per JPEG quality over ``paths``."""
    ratios: Dict[int, List[float]] = {q: [] for q in qualities}
    for p in paths:
        src = p.read_bytes()
        with PILImage.open(io.BytesIO(src)) as im:
            rgb = im.convert("RGB")
            for q in ratios:
                buf = io.BytesIO()
                rgb.save(buf, format="jpeg", quality=q, optimize=True)
                ratios[q].append(min(1.0, len(buf.getvalue()) / len(src)))
    return {q: float(np.mean(r)) for q, r in ratios.items() if r}


@click.command()
@click.option("--t4", "t4_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--gf-root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None)
@click.option("--ncore", "ncore_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None)
@click.option(
    "--camera-id",
    "camera_ids",
    multiple=True,
    help="Cameras the consumer uses (default: gaussian_factory's 5-camera ring).",
    default=("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"),
)
@click.option("--probe-jpeg-quality", "jpeg_qualities", multiple=True, type=click.IntRange(1, 100))
@click.option("--probe-images", type=int, default=20, show_default=True, help="Images sampled for the JPEG probe")
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), default=None)
def main(
    t4_dir: Path,
    gf_root: Optional[Path],
    ncore_dir: Optional[Path],
    camera_ids: tuple[str, ...],
    jpeg_qualities: tuple[int, ...],
    probe_images: int,
    json_out: Optional[Path],
) -> None:
    tables = load_annotation_tables(t4_dir / "annotation")
    sensors = index_by_token(tables["sensor"])
    calibs = index_by_token(tables["calibrated_sensor"])

    # per-channel file sizes split into keyframe / non-keyframe
    per_channel: Dict[str, Dict[str, int]] = {}
    referenced: set[str] = set()
    for sd in tables["sample_data"]:
        ch = sensors[calibs[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        size = _du(t4_dir / sd["filename"])
        referenced.add(sd["filename"])
        entry = per_channel.setdefault(ch, {"key": 0, "nonkey": 0, "n_key": 0, "n_nonkey": 0})
        kind = "key" if sd.get("is_key_frame") else "nonkey"
        entry[kind] += size
        entry[f"n_{kind}"] += 1

    data_total = _du(t4_dir / "data")
    unreferenced = data_total - sum(e["key"] + e["nonkey"] for e in per_channel.values())
    parts = {
        "input_bag (rosbag)": _du(t4_dir / "input_bag"),
        "annotation": _du(t4_dir / "annotation"),
        "map": _du(t4_dir / "map"),
        "data (unreferenced files)": max(unreferenced, 0),
    }
    other = _du(t4_dir) - data_total - sum(v for k, v in parts.items() if k != "data (unreferenced files)")
    parts["other"] = max(other, 0)
    t4_total = _du(t4_dir)

    needed = {"annotation": parts["annotation"]}
    lanelet2 = t4_dir / "map" / "lanelet2_map.osm"
    needed["map/lanelet2_map.osm"] = _du(lanelet2)
    for ch, e in per_channel.items():
        if ch in camera_ids or ch == "LIDAR_CONCAT":
            needed[f"{ch} keyframes"] = e["key"]

    print(f"T4 sequence: {t4_dir}  total {_fmt(t4_total).strip()}")
    for k, v in parts.items():
        print(_row(k, v, t4_total))
    for ch in sorted(per_channel):
        e = per_channel[ch]
        use = "used" if (ch in camera_ids or ch == "LIDAR_CONCAT") else "unused"
        print(_row(f"data/{ch} keyframes ({e['n_key']}) [{use}]", e["key"], t4_total))
        print(_row(f"data/{ch} non-keyframes ({e['n_nonkey']}) [unused]", e["nonkey"], t4_total))
    needed_total = sum(needed.values())
    print(_row("=> needed by gaussian_factory", needed_total, t4_total))

    report: Dict[str, object] = {
        "t4_total": t4_total,
        "t4_parts": parts,
        "t4_channels": per_channel,
        "needed": needed,
        "needed_total": needed_total,
    }

    if jpeg_qualities:
        imgs = sorted(
            t4_dir / sd["filename"]
            for sd in tables["sample_data"]
            if sd.get("is_key_frame")
            and sensors[calibs[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"] in camera_ids
        )
        step = max(1, len(imgs) // max(probe_images, 1))
        probe = _probe_jpeg(imgs[::step][:probe_images], jpeg_qualities)
        report["jpeg_probe"] = probe
        for q, r in probe.items():
            print(f"  JPEG re-encode q={q}: {r:.3f} x source size")

    if gf_root is not None:
        gf_total = _du(gf_root)
        print(f"\ngaussian_factory output: {gf_root}  total {_fmt(gf_total).strip()}")
        gf_layers = {}
        for name, layer in GF_LAYERS.items():
            size = _du(gf_root / layer.default_path)
            if size:
                gf_layers[name] = size
                print(_row(f"{name} ({layer.default_path})", size, gf_total))
        report["gf_total"] = gf_total
        report["gf_layers"] = gf_layers

    if ncore_dir is not None:
        stores = sorted(p for p in ncore_dir.iterdir() if p.name.endswith((".itar", ".zarr", ".json")))
        nc_total = sum(_du(p) for p in stores)
        print(f"\nNCore sequence: {ncore_dir}  total {_fmt(nc_total).strip()}")
        for p in stores:
            print(_row(p.name, _du(p), nc_total))
        print(_row("NCore / T4 total", nc_total, t4_total))
        report["ncore_total"] = nc_total
        report["ncore_stores"] = {p.name: _du(p) for p in stores}

    if json_out is not None:
        json_out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
