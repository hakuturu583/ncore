"""Minimal PCD reader used by the T4 converter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ParsedPcdField:
    """One parsed non-XYZ field from a PCD file."""

    name: str
    values: np.ndarray


@dataclass(slots=True)
class ParsedPcd:
    """Parsed PCD payload."""

    xyz: np.ndarray
    fields: list[ParsedPcdField]
    header: dict[str, Any]


def _parse_header_value(key: str, value: str) -> Any:
    if key in {"FIELDS", "TYPE"}:
        return value.split()
    if key in {"SIZE", "COUNT", "WIDTH", "HEIGHT", "POINTS"}:
        return [int(part) for part in value.split()] if key in {"SIZE", "COUNT"} else int(value)
    if key == "VIEWPOINT":
        return [float(part) for part in value.split()]
    return value


def _dtype_from_pcd_type(type_name: str, size: int) -> np.dtype:
    type_name = type_name.upper()
    if type_name == "F":
        return np.dtype(f"<f{size}")
    if type_name == "I":
        return np.dtype(f"<i{size}")
    if type_name == "U":
        return np.dtype(f"<u{size}")
    raise ValueError(f"Unsupported PCD field type {type_name!r}")


def _expand_field_values(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if values.dtype == dtype:
        return values
    return values.astype(dtype, copy=False)


def load_pcd(path: str | Path) -> ParsedPcd:
    """Load a PCD file with `ascii` or `binary` payloads."""

    pcd_path = Path(path)
    with pcd_path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected end of file while reading PCD header: {pcd_path}")
            decoded = line.decode("utf-8", errors="replace").strip()
            header_lines.append(decoded)
            if decoded.upper().startswith("DATA "):
                data_encoding = decoded.split(maxsplit=1)[1].lower()
                break

        header: dict[str, Any] = {}
        for line in header_lines:
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            key, value = parts
            header[key.upper()] = _parse_header_value(key.upper(), value)

        fields = list(header.get("FIELDS", []))
        sizes = list(header.get("SIZE", []))
        types = list(header.get("TYPE", []))
        counts = list(header.get("COUNT", [])) or [1] * len(fields)
        if len(counts) != len(fields):
            raise ValueError(f"PCD COUNT length mismatch in {pcd_path}")
        if not fields or len(sizes) != len(fields) or len(types) != len(fields):
            raise ValueError(f"Malformed PCD header in {pcd_path}")
        if "POINTS" not in header:
            width = int(header.get("WIDTH", 0))
            height = int(header.get("HEIGHT", 1))
            header["POINTS"] = width * height

        points = int(header["POINTS"])
        total_scalar_columns = int(sum(counts))

        if data_encoding == "ascii":
            data_text = f.read().decode("utf-8", errors="replace")
            flat = np.fromstring(data_text, sep=" ", dtype=np.float64, count=points * total_scalar_columns)
            if flat.size != points * total_scalar_columns:
                raise ValueError(f"PCD ascii payload size mismatch in {pcd_path}")
            matrix = flat.reshape(points, total_scalar_columns)
            cursor = 0
            xyz_columns: dict[str, np.ndarray] = {}
            extra_fields: list[ParsedPcdField] = []
            for name, size, type_name, count in zip(fields, sizes, types, counts):
                dtype = _dtype_from_pcd_type(type_name, size)
                slice_values = matrix[:, cursor : cursor + count]
                cursor += count
                if count == 1:
                    values = _expand_field_values(slice_values[:, 0], dtype)
                else:
                    values = _expand_field_values(slice_values, dtype)
                if name in {"x", "y", "z"}:
                    xyz_columns[name] = values.astype(np.float32, copy=False)
                else:
                    extra_fields.append(ParsedPcdField(name=name, values=values))
        elif data_encoding == "binary":
            scalar_dtype_fields: list[tuple[str, np.dtype | tuple[np.dtype, tuple[int, ...]]]] = []
            for name, size, type_name, count in zip(fields, sizes, types, counts):
                dtype = _dtype_from_pcd_type(type_name, size)
                if count == 1:
                    scalar_dtype_fields.append((name, dtype))
                else:
                    scalar_dtype_fields.append((name, dtype, (count,)))
            structured_dtype = np.dtype(scalar_dtype_fields)
            raw = f.read(points * structured_dtype.itemsize)
            if len(raw) != points * structured_dtype.itemsize:
                raise ValueError(f"PCD binary payload size mismatch in {pcd_path}")
            structured = np.frombuffer(raw, dtype=structured_dtype, count=points)
            xyz_columns = {}
            extra_fields = []
            for name in fields:
                values = np.array(structured[name], copy=False)
                if name in {"x", "y", "z"}:
                    xyz_columns[name] = values.astype(np.float32, copy=False)
                else:
                    extra_fields.append(ParsedPcdField(name=name, values=values))
        else:
            raise NotImplementedError(f"Unsupported PCD DATA encoding {data_encoding!r} in {pcd_path}")

    if set(xyz_columns) != {"x", "y", "z"}:
        raise ValueError(f"PCD file {pcd_path} must contain x/y/z fields")

    xyz = np.stack([xyz_columns["x"], xyz_columns["y"], xyz_columns["z"]], axis=1).astype(np.float32, copy=False)
    return ParsedPcd(xyz=xyz, fields=extra_fields, header=header)
