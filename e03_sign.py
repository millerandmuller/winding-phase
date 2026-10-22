"""E-03: give each ladder the fitter's winding sign from a prior fit grid.

    python e03_sign.py out/e03_ladders_z8528_three_b0.35_full.json --grid grids_PHerc0826_A0.npz \
        --gold gold0826_heldout8.json --out out/

A ladder from e03_ladders.py counts windings along the outward normal; where
a crushed sheet doubles back, the fitter's winding decreases outward and the
sign is wrong. No local measurement can tell. This step reads the direction
of increasing winding from a prior fit (PCU's tracks-only fit A0, or any
fit output stacked as grids_<run>.npz: one (3, rows, cols) array per winding
wNNN with x, y, z in level-0 voxels, rows along z, -1 for missing) and
reverses the ladders that run against it. Ladders with no readable sign keep
the outward sign (--unsigned keep) or are dropped (--unsigned drop).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from e02a_baseline import load_ladders
from e03_ladders import score_against_gold


def grid_curves_at_z(grid: np.lib.npyio.NpzFile, z0: float, upsample: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """(N, 2) level-0 (y, x) points on every winding surface at height z0, with their winding index.

    Each winding's grid rows run along z; the two rows bracketing z0 are
    interpolated to z0 and the columns upsampled linearly."""
    pts, winds = [], []
    for key in grid.files:
        g = grid[key]  # (3, rows, cols): x, y, z
        w = int(key[1:])
        z = np.where(g[2] > 0, g[2], np.nan)
        with np.errstate(all="ignore"):
            zrow = np.nanmean(z, axis=1)
        ok = np.isfinite(zrow)
        if ok.sum() < 2:
            continue
        # The band's first and last slices can sit a fraction of a row outside
        # the grid's row means; clamp within one row spacing, skip beyond.
        zc = float(np.clip(z0, np.nanmin(zrow), np.nanmax(zrow)))
        if abs(zc - z0) > 20:
            continue
        rows = np.flatnonzero(ok)
        k = min(np.searchsorted(zrow[rows], zc), len(rows) - 1)
        i1 = rows[k]
        i0 = rows[max(0, k - 1)]
        t = 0.0 if i0 == i1 else (zc - zrow[i0]) / (zrow[i1] - zrow[i0])
        x = (1 - t) * g[0, i0] + t * g[0, i1]
        y = (1 - t) * g[1, i0] + t * g[1, i1]
        valid = (g[0, i0] > 0) & (g[0, i1] > 0)
        if valid.sum() < 2:
            continue
        cols = np.flatnonzero(valid)
        fine = np.linspace(cols[0], cols[-1], (cols[-1] - cols[0]) * upsample + 1)
        xf = np.interp(fine, cols, x[cols]); yf = np.interp(fine, cols, y[cols])
        pts.append(np.stack([yf, xf], axis=1)); winds.append(np.full(len(fine), w))
    return np.concatenate(pts), np.concatenate(winds)


def ladder_sign_gradient(lad_yx_l0: np.ndarray, tree: cKDTree, pts: np.ndarray, winds: np.ndarray, radius: float) -> int:
    """Sign of the fitted winding gradient along the ladder, from a plane fit
    w ~ a + b y + c x over grid points within `radius` of the ladder midpoint.
    A fit with many slips still has a smooth large-scale gradient."""
    mid = lad_yx_l0.mean(axis=0)
    idx = tree.query_ball_point(mid, radius)
    if len(idx) < 12 or len(np.unique(winds[idx])) < 2:
        return 0
    A = np.column_stack([np.ones(len(idx)), pts[idx] - mid])
    coef, *_ = np.linalg.lstsq(A, winds[idx].astype(float), rcond=None)
    along = lad_yx_l0[-1] - lad_yx_l0[0]
    s = np.sign(float(coef[1] * along[0] + coef[2] * along[1]))
    return int(s)


def ladder_sign(lad_yx_l0: np.ndarray, tree: cKDTree, winds: np.ndarray, radius: float) -> int:
    """+1 if the fitted winding grows along the ladder, -1 if it shrinks, 0 if unreadable."""
    d, idx = tree.query(lad_yx_l0, distance_upper_bound=radius)
    w = np.where(np.isfinite(d), winds[np.minimum(idx, len(winds) - 1)], np.nan)
    dw = np.diff(w)
    dw = dw[np.isfinite(dw) & (dw != 0)]
    if len(dw) == 0:
        return 0
    s = np.sign(np.median(dw))
    return int(s) if s != 0 else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ladders", type=Path)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--gold", type=Path, default=None)
    ap.add_argument("--method", choices=["gradient", "nearest"], default="gradient")
    ap.add_argument("--radius", type=float, default=120.0, help="level-0 voxels: neighbourhood of the plane fit (gradient) or the nearest-surface cutoff (nearest)")
    ap.add_argument("--unsigned", choices=["keep", "drop"], default="keep")
    ap.add_argument("--match", type=float, default=2.5)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    d = json.loads(args.ladders.read_text())
    cols = d["collections"]
    z0 = float(next(iter(cols.values()))["points"]["1"]["p"][2])
    pts, winds = grid_curves_at_z(np.load(args.grid), z0)
    tree = cKDTree(pts)

    out_cols, counts = {}, {"kept": 0, "flipped": 0, "unreadable": 0, "dropped": 0}
    signed_l1 = []
    for i, (key, c) in enumerate(cols.items()):
        key = str(int(z0) * 100000 + i + 1)  # the fitter parses collection keys as integers
        points = [p for _, p in sorted(c["points"].items(), key=lambda kv: int(kv[0]))]
        yx0 = np.array([[p["p"][1], p["p"][0]] for p in points])
        s = (ladder_sign_gradient(yx0, tree, pts, winds, args.radius) if args.method == "gradient"
             else ladder_sign(yx0, tree, winds, args.radius))
        if s == 0:
            counts["unreadable"] += 1
            if args.unsigned == "drop":
                counts["dropped"] += 1
                continue
        elif s < 0:
            counts["flipped"] += 1
            points = points[::-1]
        else:
            counts["kept"] += 1
        meta = dict(c["metadata"]); meta["sign"] = {1: "fit grid", -1: "fit grid (reversed)", 0: "outward normal (no grid)"}[s]
        meta["sign_grid"] = args.grid.name
        out_cols[key] = {"name": c["name"], "metadata": meta,
                         "points": {str(j + 1): {"p": p["p"], "wind_a": float(j + 1)} for j, p in enumerate(points)}}
        signed_l1.append(np.array([[p["p"][1] / 2.0, p["p"][0] / 2.0] for p in points]))

    stats = {"ladders_in": len(cols), "z0": z0, "grid": args.grid.name, "method": args.method, "radius": args.radius,
             "grid_points_at_z": int(len(pts)), **counts}
    if args.gold:
        gold = load_ladders(args.gold, int(z0), 0, target_level=1)
        stats["against_gold"] = score_against_gold(signed_l1, gold, args.match, args.kmax)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.ladders.stem.replace("e03_ladders_", "")
    (args.out / f"e03_signed_{stem}.json").write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": out_cols}))
    (args.out / f"e03_signstats_{stem}.json").write_text(json.dumps(stats, indent=2))
    g = stats.get("against_gold", {})
    print(stem, {k: stats[k] for k in ("ladders_in", "kept", "flipped", "unreadable")}, "| tests", g.get("tests"), "signed", g.get("correct"), "unsigned", g.get("correct_unsigned"))


if __name__ == "__main__":
    main()
