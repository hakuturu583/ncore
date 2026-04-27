import pathlib

import numpy as np

from tools.data_converter.t4.pcd import load_pcd


def _write_pcd(path: pathlib.Path, header: str, payload: bytes) -> None:
    path.write_bytes(header.encode("utf-8") + payload)


def test_load_ascii_pcd(tmp_path: pathlib.Path) -> None:
    pcd_path = tmp_path / "map_ascii.pcd"
    header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity ring
SIZE 4 4 4 4 2
TYPE F F F F U
COUNT 1 1 1 1 1
WIDTH 2
HEIGHT 1
POINTS 2
DATA ascii
"""
    payload = b"1.0 2.0 3.0 0.5 7\n4.0 5.0 6.0 0.25 8\n"
    _write_pcd(pcd_path, header, payload)

    parsed = load_pcd(pcd_path)

    assert parsed.xyz.dtype == np.float32
    assert parsed.xyz.shape == (2, 3)
    np.testing.assert_allclose(parsed.xyz, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    assert [field.name for field in parsed.fields] == ["intensity", "ring"]
    np.testing.assert_allclose(parsed.fields[0].values, np.array([0.5, 0.25], dtype=np.float32))
    np.testing.assert_array_equal(parsed.fields[1].values, np.array([7, 8], dtype=np.uint16))


def test_load_binary_pcd(tmp_path: pathlib.Path) -> None:
    pcd_path = tmp_path / "map_binary.pcd"
    header = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 2
HEIGHT 1
POINTS 2
DATA binary
"""
    structured = np.array(
        [(1.0, 2.0, 3.0, 0.1), (4.0, 5.0, 6.0, 0.2)],
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]),
    )
    _write_pcd(pcd_path, header, structured.tobytes())

    parsed = load_pcd(pcd_path)

    assert parsed.xyz.dtype == np.float32
    np.testing.assert_allclose(parsed.xyz, np.column_stack([structured["x"], structured["y"], structured["z"]]))
    assert [field.name for field in parsed.fields] == ["rgb"]
    np.testing.assert_allclose(parsed.fields[0].values, structured["rgb"])
