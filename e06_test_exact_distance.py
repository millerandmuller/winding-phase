"""Synthetic check of e06_exact_distance.py (validation 1 of E-06).

    python e06_test_exact_distance.py

A cylinder section (radius 300, vertex spacing 22) with a 3x3 hole of invalid vertices; 1,000 points at
distances 0 to 30 along the surface normal, seed 1. The exact measure is compared with an independent
reference: for every triangle, the distance to its plane if the projection falls inside, otherwise the
nearest of its three edges as segments. Passes when the largest difference is at most 0.01 voxel. Also
prints the difference to the smooth cylinder (the chord error, about spacing^2 / (8 * radius)).
"""

from __future__ import annotations

import sys

import numpy as np

from e06_exact_distance import ExactSurface, triangulate

TOLERANCE = 0.01


def cylinder_grid(radius: float = 300.0, spacing: float = 22.0, rows: int = 30, cols: int = 30) -> np.ndarray:
    """Grid (3, rows, cols) of x, y, z on a cylinder around the z axis, offset so x stays positive."""
    theta = np.arange(cols) * spacing / radius
    z = 1000.0 + np.arange(rows) * spacing
    T, Z = np.meshgrid(theta, z)
    g = np.stack([2000.0 + radius * np.cos(T), 2000.0 + radius * np.sin(T), Z])
    g[:, 13:16, 13:16] = 0.0  # 3x3 hole of invalid vertices (x <= 0)
    return g


def segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def reference_distance(p: np.ndarray, tri: np.ndarray) -> float:
    """Plane distance when the projection is inside (barycentric test), else the nearest edge."""
    a, b, c = tri
    n = np.cross(b - a, c - a)
    n = n / np.linalg.norm(n)
    q = p - np.dot(p - a, n) * n
    v0, v1, v2 = b - a, c - a, q - a
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    if v >= 0 and w >= 0 and v + w <= 1:
        return abs(float(np.dot(p - a, n)))
    return min(segment_distance(p, a, b), segment_distance(p, b, c), segment_distance(p, c, a))


def main() -> int:
    g = cylinder_grid()
    tri = triangulate({"w001": g}, 900, 1800)
    surf = ExactSurface(tri)
    rng = np.random.default_rng(1)
    n = 1000
    u, v = rng.uniform(1, 28, n), rng.uniform(1, 28, n)  # column and row position in cells
    theta = u * 22.0 / 300.0
    z = 1000.0 + v * 22.0
    near_hole = (u > 11) & (u < 18) & (v > 11) & (v < 18)
    dist = rng.uniform(0, 30, n)
    side = rng.choice([-1.0, 1.0], n)
    r = 300.0 + side * dist
    pts = np.stack([2000.0 + r * np.cos(theta), 2000.0 + r * np.sin(theta), z], 1)

    worst, worst_cyl, unassigned = 0.0, 0.0, 0
    for p, d_true, hole in zip(pts, dist, near_hole):
        d_x, w_x, _ = surf.nearest(p)
        d_ref = min(reference_distance(p, t) for t in tri["corners"])
        if d_ref <= 40.0:
            worst = max(worst, abs(d_x - d_ref))
        if w_x is None:
            unassigned += 1
            continue
        if not hole:
            worst_cyl = max(worst_cyl, abs(d_x - d_true))
    print(f"triangles {len(tri['corners'])}, points {n}, largest difference to reference {worst:.2e}, "
          f"largest difference to the smooth cylinder {worst_cyl:.3f} (outside the hole), none in reach {unassigned}")
    ok = worst <= TOLERANCE
    print("SYNTHETIC: PASS" if ok else "SYNTHETIC: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
