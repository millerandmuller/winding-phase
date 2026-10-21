"""Umbilicus (scroll axis) lookup: the outward reference direction for one slice.

The winding coordinate grows outward from the umbilicus, so the radial
direction from it orients the in-plane normals, which the Lasagna product
does not document as oriented. Control points are in level-0 volume
coordinates; `axis_at` interpolates linearly in z and rescales to a pyramid
level.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_control_points(path: Path) -> np.ndarray:
    """Return an (N, 3) float array of (z, y, x) at level 0, sorted by z."""
    pts = json.loads(Path(path).read_text())["control_points"]
    arr = np.array([(p["z"], p["y"], p["x"]) for p in pts], dtype=np.float64)
    return arr[np.argsort(arr[:, 0])]


def axis_at(control_points: np.ndarray, z: int, level: int) -> tuple[float, float]:
    """(y, x) of the umbilicus at pyramid `level` for level-`level` slice index z."""
    scale = float(2**level)
    z0 = z * scale
    zs, ys, xs = control_points[:, 0], control_points[:, 1], control_points[:, 2]
    y = float(np.interp(z0, zs, ys)) / scale
    x = float(np.interp(z0, zs, xs)) / scale
    return y, x


def orient_normals(
    nx: np.ndarray, ny: np.ndarray, y0: int, x0: int, axis_yx: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip each normal so that it points away from the umbilicus.

    Returns (nx, ny, radial_dot): radial_dot is |n . r_hat| after the flip,
    near zero where the sheet runs radially and the orientation is unreliable.
    """
    yy, xx = np.mgrid[0 : nx.shape[0], 0 : nx.shape[1]]
    ry = (yy + y0) - axis_yx[0]
    rx = (xx + x0) - axis_yx[1]
    rn = np.hypot(ry, rx) + 1e-6
    dot = (nx * rx + ny * ry) / rn
    sign = np.where(dot < 0, -1.0, 1.0).astype(np.float32)
    return nx * sign, ny * sign, np.abs(dot).astype(np.float32)
