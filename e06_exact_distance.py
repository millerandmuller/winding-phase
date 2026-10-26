"""Exact distance from human label points to a fitted spiral grid, beside PCU e31b's measure as written.

    python e06_exact_distance.py OURS=grids_OURS.npz A0=grids_A0.npz --labels abs_winding.json relative_windings.json \
        --z0 16900 --z1 17900 --out points_B7.json --counts counts_B7_exact.json

For each winding grid, every 2x2 cell whose four corners are valid (x > 0) and which has a corner within
z0 - 60 to z1 + 60 is split into two triangles along its (a, b) to (a+1, b+1) diagonal. A label point's
distance is the exact Euclidean distance to the nearest such triangle of any winding (every triangle whose
centroid is within 40 voxels plus its circumradius is checked); its winding is that triangle's winding. A
point is assigned when the distance is under 10 voxels, e31b's cut-off.

Ladders are selected as e31b selects them (`e05_fallback_check.load_ladders`), and the as-written e31b
distance and winding come from the copy of `winding_at` in `e05_fallback_check.py`, so both measures see
the same points in the same order. `--out` writes one record per fit and label point; `--counts` writes
per-ladder assigned pairs and slips under the exact measure, in the format `e03_pool_bands.py` reads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from e05_fallback_check import build_index, load_ladders, parse_fit, winding_at

CUTOFF = 10.0
SEARCH = 40.0
Z_MARGIN = 60


def closest_points_on_triangles(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Closest point on triangle (a, b, c) to p, row by row (Ericson, Real-Time Collision Detection, 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    # Interior first, then overwrite with the edge and vertex regions that apply.
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v_in, w_in = vb / denom, vc / denom
        out = a + v_in[:, None] * ab + w_in[:, None] * ac
        bc_den = (d4 - d3) + (d5 - d6)
        on_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
        t = (d4 - d3) / bc_den
        out = np.where(on_bc[:, None], b + t[:, None] * (c - b), out)
        on_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t = d2 / (d2 - d6)
        out = np.where(on_ac[:, None], a + t[:, None] * ac, out)
        out = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, out)
        on_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = d1 / (d1 - d3)
        out = np.where(on_ab[:, None], a + t[:, None] * ab, out)
    out = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, out)
    out = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, out)
    return out


def triangulate(G: dict, z0: float, z1: float) -> dict:
    """Triangles of all valid cells near the band: corners (n, 3, 3), winding, cell index, invalid-neighbour flag."""
    corners, winding, cell, border = [], [], [], []
    for k, g in G.items():
        valid = g[0] > 0
        cell_ok = valid[:-1, :-1] & valid[1:, :-1] & valid[1:, 1:] & valid[:-1, 1:]
        z = g[2]
        z_lo = np.minimum.reduce([z[:-1, :-1], z[1:, :-1], z[1:, 1:], z[:-1, 1:]])
        z_hi = np.maximum.reduce([z[:-1, :-1], z[1:, :-1], z[1:, 1:], z[:-1, 1:]])
        cell_ok &= (z_hi > z0 - Z_MARGIN) & (z_lo < z1 + Z_MARGIN)
        aa, bb = np.nonzero(cell_ok)
        if not len(aa):
            continue
        P = np.moveaxis(g[:3], 0, -1).astype(float)
        A, B, C, D = P[aa, bb], P[aa + 1, bb], P[aa + 1, bb + 1], P[aa, bb + 1]
        corners += [np.stack([A, B, C], 1), np.stack([A, C, D], 1)]
        w = np.full(len(aa), int(k[1:]))
        winding += [w, w]
        ab = np.stack([aa, bb], 1)
        cell += [ab, ab]
        brd = invalid_nearby(valid, aa, bb)
        border += [brd, brd]
    return {"corners": np.concatenate(corners), "winding": np.concatenate(winding),
            "cell": np.concatenate(cell), "border": np.concatenate(border)}


def invalid_nearby(valid: np.ndarray, aa: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """True where a vertex in rows a-1..a+2, columns b-1..b+2 is invalid or off the grid."""
    pad = np.pad(valid, 1, constant_values=False)
    out = np.zeros(len(aa), bool)
    for da in range(4):
        for db in range(4):
            out |= ~pad[aa + da, bb + db]
    return out


class ExactSurface:
    """Nearest triangle of a triangulated fit, with a centroid KD-tree for candidate search."""

    def __init__(self, tri: dict):
        self.tri = tri
        c = tri["corners"]
        self.centroid = c.mean(1)
        self.radius = np.linalg.norm(c - self.centroid[:, None], axis=2).max(1)
        self.tree = cKDTree(self.centroid)
        self.max_radius = float(self.radius.max()) if len(self.radius) else 0.0

    def nearest(self, p: np.ndarray) -> tuple[float, int | None, int | None]:
        """Exact distance to the nearest triangle within reach, its winding and its triangle index."""
        idx = np.array(self.tree.query_ball_point(p, SEARCH + self.max_radius), int)
        if not len(idx):
            return np.inf, None, None
        idx = idx[np.linalg.norm(self.centroid[idx] - p, axis=1) <= SEARCH + self.radius[idx]]
        if not len(idx):
            return np.inf, None, None
        c = self.tri["corners"][idx]
        q = closest_points_on_triangles(np.broadcast_to(p, (len(idx), 3)), c[:, 0], c[:, 1], c[:, 2])
        d = np.linalg.norm(q - p, axis=1)
        i = int(np.argmin(d))
        return float(d[i]), int(self.tri["winding"][idx[i]]), int(idx[i])


def pair_counts(lad: list, windings: list[np.ndarray]) -> tuple[list[int], list[int]]:
    """Per-ladder assigned pairs and slips under e31b's rule (sign from the median step).

    Intentional duplicate of the counting inside e05_fallback_check.score: that file stays a frozen copy of
    PCU's code, and here the windings come from another measure.
    """
    assigned, slips = [], []
    for (_, w), v in zip(lad, windings):
        dv, dw = np.diff(v), np.diff(w)
        ok = np.isfinite(dv)
        if not ok.any():
            assigned.append(0)
            slips.append(0)
            continue
        sg = np.sign(np.nanmedian(dv)) or 1
        slips.append(int(np.sum(sg * dv[ok] != dw[ok])))
        assigned.append(int(ok.sum()))
    return assigned, slips


def measure_fit(path: Path, lad: list, z0: int, z1: int) -> list[dict]:
    """Per label point: e31b as written (distance, winding) and the exact measure (distance, winding, border)."""
    with np.load(path) as npz:
        G = {k: npz[k] for k in npz.files}
    V, Wv, tree = build_index(G, z0, z1)
    surf = ExactSurface(triangulate(G, z0, z1))
    rows = []
    for li, (pts, _) in enumerate(lad):
        for pi, p in enumerate(pts):
            d_e, w_e = winding_at(p, G, V, Wv, tree, False)
            d_x, w_x, t = surf.nearest(p)
            rows.append({"ladder": li, "point": pi,
                         "e31b_d": float(d_e), "e31b_w": None if w_e is None else int(w_e),
                         "exact_d": d_x, "exact_w": w_x,
                         "exact_border": None if t is None else bool(surf.tri["border"][t])})
    return rows


def exact_windings(lad: list, rows: list[dict]) -> list[np.ndarray]:
    """Per ladder, the exact-measure winding of each point, NaN where unassigned."""
    by = {(r["ladder"], r["point"]): r for r in rows}
    out = []
    for li, (pts, _) in enumerate(lad):
        v = [by[(li, pi)]["exact_w"] if by[(li, pi)]["exact_d"] < CUTOFF else np.nan for pi in range(len(pts))]
        out.append(np.array(v, float))
    return out


def json_safe(rows: list[dict]) -> list[dict]:
    return [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fits", nargs="+", help="NAME=grids.npz (or a bare path), one or more fits of the same band")
    ap.add_argument("--labels", type=Path, nargs="+", required=True, help="abs_winding.json relative_windings.json")
    ap.add_argument("--z0", type=int, required=True)
    ap.add_argument("--z1", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True, help="per-point records per fit (JSON)")
    ap.add_argument("--counts", type=Path, default=None, help="per-ladder counts under the exact measure")
    args = ap.parse_args()
    lad = load_ladders(args.labels, args.z0, args.z1)
    points = {"z": [args.z0, args.z1], "ladders": [[[float(x) for x in p] for p in pts] for pts, _ in lad], "fits": {}}
    counts = {"per_ladder_pairs": [len(w) - 1 for _, w in lad]}
    for spec in args.fits:
        name, path = parse_fit(spec)
        rows = measure_fit(path, lad, args.z0, args.z1)
        points["fits"][name] = json_safe(rows)
        a, s = pair_counts(lad, exact_windings(lad, rows))
        counts[name] = {"per_ladder_assigned": a, "per_ladder_slips": s}
        n_e = sum(r["e31b_d"] < CUTOFF for r in rows)
        n_x = sum(r["exact_d"] < CUTOFF for r in rows)
        print(f"{name} z {args.z0}-{args.z1}: points {len(rows)}, e31b assigned {n_e}, exact assigned {n_x}, "
              f"pairs exact {sum(a)}/{sum(counts['per_ladder_pairs'])}, slips {sum(s)}", flush=True)
    args.out.write_text(json.dumps(points))
    if args.counts:
        args.counts.write_text(json.dumps(counts))


if __name__ == "__main__":
    main()
