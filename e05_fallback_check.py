"""PCU e31b's human-ladder scoring with and without its raw-vertex fallback, on one or more fit grids.

    python e05_fallback_check.py OURS=grids_OURS.npz A0=grids_A0.npz --labels abs_winding.json relative_windings.json \
        --z0 16900 --z1 17900 --out totals.json --counts band

`winding_at` is copied from PCU's exp/e31b_bootstrap.py (Jashann/vesuvius-scrolling, 4f4997a, lines 59-73).
As written, if any vertex of the 5x5 patch around the nearest vertex is invalid, it takes the distance to
the nearest raw vertex. With --fixed behaviour (always computed, printed second), it also upsamples every
2x2 grid cell of the patch whose four corners are valid, and keeps the raw-vertex distance only if that is
smaller. Ladder selection and slip counting are e31b's (wind_a points sorted, z spread <= 2, mean z in
[z0 + 20, z1 - 20), sign per ladder from the median step, 10-voxel cut-off).

The copied parts (load_ladders, build_index, winding_at, score) are from vesuvius-scrolling
(https://github.com/Jashann/vesuvius-scrolling), under this licence:

    MIT License

    Copyright (c) 2026 Jashanjot Singh Gill and Nhat Nam Tran

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def load_ladders(paths: list[Path], z0: int, z1: int) -> list[tuple[np.ndarray, np.ndarray]]:
    lad = []
    for path in paths:
        for col in json.loads(path.read_text())["collections"].values():
            P = sorted([p for p in col["points"].values() if p.get("wind_a") is not None], key=lambda p: p["wind_a"])
            if len(P) < 2:
                continue
            zz = np.array([p["p"][2] for p in P])
            if np.ptp(zz) > 2 or not (z0 + 20 <= zz.mean() < z1 - 20):
                continue
            lad.append((np.array([p["p"] for p in P], float), np.array([p["wind_a"] for p in P], float)))
    return lad


def build_index(G: dict, z0: int, z1: int):
    """All valid grid vertices in the band (with margin), their winding and grid index, and a KD-tree."""
    V, Wv = [], []
    for k, g in G.items():
        ok = (g[0] > 0) & (g[2] > z0 - 60) & (g[2] < z1 + 60)
        ii, jj = np.nonzero(ok)
        V.append(np.stack([g[0][ok], g[1][ok], g[2][ok]], 1))
        Wv.append(np.stack([np.full(len(ii), int(k[1:])), ii, jj], 1))
    V = np.concatenate(V)
    return V, np.concatenate(Wv), cKDTree(V)


def valid_cell_distance(sub: np.ndarray, p: np.ndarray) -> float:
    """Smallest distance from p to the upsampled 2x2 cells of sub whose four corners are valid."""
    valid = sub[0] > 0
    d = np.inf
    for a in range(sub.shape[1] - 1):
        for b in range(sub.shape[2] - 1):
            if valid[a:a + 2, b:b + 2].all():
                cell = np.stack([ndi.zoom(sub[q, a:a + 2, b:b + 2], 6, order=1) for q in range(3)])
                d = min(d, float(np.min(np.linalg.norm(cell.reshape(3, -1).T - p, axis=1))))
    return d


def winding_at(p, G, V, Wv, tree, fixed: bool):
    idx = tree.query_ball_point(p, 40)
    best = (np.inf, None)
    for w in np.unique(Wv[idx, 0]):
        sel = [i for i in idx if Wv[i, 0] == w]
        i0 = sel[int(np.argmin(np.linalg.norm(V[sel] - p, axis=1)))]
        g = G[f"w{w:03d}"]
        r, c = Wv[i0, 1], Wv[i0, 2]
        sub = g[:, max(r - 2, 0):min(r + 3, g.shape[1]), max(c - 2, 0):min(c + 3, g.shape[2])]
        if (sub[0] <= 0).any():
            d = np.min(np.linalg.norm(V[sel] - p, axis=1))
            if fixed:
                d = min(d, valid_cell_distance(sub, p))
        else:
            up = np.stack([ndi.zoom(sub[q], 6, order=1) for q in range(3)])
            d = np.min(np.linalg.norm(up.reshape(3, -1).T - p, axis=1))
        if d < best[0]:
            best = (d, w)
    return best


def score(lad, G, V, Wv, tree, fixed: bool) -> dict:
    """Totals and per-ladder counts (assigned pairs, slips) for one grid under one fallback rule."""
    per_assigned, per_slips, points = [], [], 0
    for pts, w in lad:
        res = [winding_at(p, G, V, Wv, tree, fixed) for p in pts]
        v = np.array([b[1] if b[1] is not None and b[0] < 10 else np.nan for b in res], float)
        points += int(np.isfinite(v).sum())
        dv, dw = np.diff(v), np.diff(w)
        ok = np.isfinite(dv)
        if not ok.any():
            per_assigned.append(0)
            per_slips.append(0)
            continue
        sg = np.sign(np.nanmedian(dv)) or 1
        per_slips.append(int(np.sum(sg * dv[ok] != dw[ok])))
        per_assigned.append(int(ok.sum()))
    pairs = sum(len(w) - 1 for _, w in lad)
    assigned, slips = sum(per_assigned), sum(per_slips)
    return {"points_assigned": points, "points": sum(len(w) for _, w in lad), "pairs_assigned": assigned,
            "pairs": pairs, "slips": slips, "correct_of_all": (assigned - slips) / pairs,
            "per_ladder_assigned": per_assigned, "per_ladder_slips": per_slips}


def parse_fit(spec: str) -> tuple[str, Path]:
    """NAME=grids.npz, or a bare path named by its file stem."""
    name, sep, path = spec.partition("=")
    return (name, Path(path)) if sep else (Path(spec).stem, Path(spec))


def score_grid(path: Path, lad, z0: int, z1: int) -> dict:
    with np.load(path) as npz:
        G = {k: npz[k] for k in npz.files}
    V, Wv, tree = build_index(G, z0, z1)
    return {"as_written": score(lad, G, V, Wv, tree, False), "valid_cells": score(lad, G, V, Wv, tree, True)}


def counts_file(lad, results: dict, mode: str) -> dict:
    """One band in e03_correct_of_all.py's per-ladder format, so e03_pool_bands.py can pool it."""
    out = {"per_ladder_pairs": [len(w) - 1 for _, w in lad]}
    for name, r in results.items():
        out[name] = {k: r[mode][k] for k in ("per_ladder_assigned", "per_ladder_slips")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fits", nargs="+", help="NAME=grids.npz (or a bare path), one or more fits of the same band")
    ap.add_argument("--labels", type=Path, nargs="+", required=True, help="abs_winding.json relative_windings.json")
    ap.add_argument("--z0", type=int, required=True)
    ap.add_argument("--z1", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None, help="totals per fit (JSON)")
    ap.add_argument("--counts", default=None, metavar="PREFIX",
                    help="also write PREFIX_as_written.json and PREFIX_valid_cells.json in e03_correct_of_all format")
    args = ap.parse_args()
    lad = load_ladders(args.labels, args.z0, args.z1)
    results = {}
    for spec in args.fits:
        name, path = parse_fit(spec)
        results[name] = score_grid(path, lad, args.z0, args.z1)
        for mode in ("as_written", "valid_cells"):
            r = results[name][mode]
            print(f"{name} z {args.z0}-{args.z1} {mode}: points {r['points_assigned']}/{r['points']}, "
                  f"pairs {r['pairs_assigned']}/{r['pairs']}, slips {r['slips']}, "
                  f"correct of all {100 * r['correct_of_all']:.1f}%", flush=True)
    if args.out:
        args.out.write_text(json.dumps({"z": [args.z0, args.z1], "ladders": len(lad), "fits": results}, indent=1))
    if args.counts:
        for mode in ("as_written", "valid_cells"):
            Path(f"{args.counts}_{mode}.json").write_text(json.dumps(counts_file(lad, results, mode)))


if __name__ == "__main__":
    main()
