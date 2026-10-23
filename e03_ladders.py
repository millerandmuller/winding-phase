"""E-03: certified winding ladders from the Lasagna product, in VC3D point-collection JSON.

    python e03_ladders.py --z0 8528 --gold gold0826_heldout8.json --out out/

Seeds sit on cosine ridges, one per grid cell. From each seed the walk steps
outward along the oriented normal to the next ridge; each hop is certified
as one winding by the integrated grad_mag density (--cert two) or by the
density and the 2D unwrap together (--cert three); hops chain into a ladder
with relative windings until a hop fails. The ridge walk proposes, the other
estimates certify, so the ridge count is not a vote here.

Output: the PCU gold format (vc_pointcollections_json_version 1, level-0
coordinates, winding_is_absolute false), one collection per ladder.

With --gold, every gold pair on the slice (rungs 1 to --kmax apart) whose two
rungs each lie within --match px of a point of one of our ladders is a test:
correct if the winding difference of that ladder equals the gold one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label, map_coordinates
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from skimage.restoration import unwrap_phase

from e02a_baseline import load_ladders
from e02a_certify import density_integral, interior_ridges
from lasagna_store import SCROLLS, level_shape, read_slice_window_cos_level1
from phase import wrapped_phase
from umbilicus import axis_at, load_control_points, orient_normals

DATA = Path(__file__).parent / "data"
TWO_PI = 2.0 * np.pi


def sample(arr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear samples of arr at (N, 2) (y, x) points."""
    return map_coordinates(arr, pts.T, order=1, mode="nearest")


def ridge_mask(cos_s: np.ndarray, nx: np.ndarray, ny: np.ndarray, min_cos: float) -> np.ndarray:
    """Pixels where the smoothed cosine is a local maximum along the normal."""
    yy, xx = np.mgrid[0 : cos_s.shape[0], 0 : cos_s.shape[1]].astype(np.float32)
    ahead = map_coordinates(cos_s, [yy + ny, xx + nx], order=1, mode="nearest")
    behind = map_coordinates(cos_s, [yy - ny, xx - nx], order=1, mode="nearest")
    return (cos_s >= ahead) & (cos_s >= behind) & (cos_s > min_cos)


def grid_seeds(score: np.ndarray, allowed: np.ndarray, stride: int) -> np.ndarray:
    """Best allowed pixel per stride x stride cell, as (N, 2) float (y, x)."""
    seeds = []
    masked = np.where(allowed, score, -np.inf)
    for y in range(0, score.shape[0], stride):
        for x in range(0, score.shape[1], stride):
            cell = masked[y : y + stride, x : x + stride]
            if not np.isfinite(cell).any():
                continue
            iy, ix = np.unravel_index(int(np.argmax(cell)), cell.shape)
            seeds.append((y + iy, x + ix))
    return np.array(seeds, dtype=np.float64).reshape(-1, 2)


def next_ridge(cos_s: np.ndarray, a: np.ndarray, n: np.ndarray, max_len: float, prominence: float,
               min_len: float = 2.0, step: float = 0.5) -> np.ndarray | None:
    """First cosine peak along the ray a + s n, min_len <= s <= max_len, or None."""
    s = np.arange(0.0, max_len + step, step)
    pts = a[None, :] + s[:, None] * n[None, :]
    prof = sample(cos_s, pts)
    peaks, _ = find_peaks(prof, prominence=prominence)
    peaks = peaks[s[peaks] >= min_len]
    return pts[peaks[0]] if len(peaks) else None


class Certifier:
    """Votes on one hop a -> b; `unwrapped` is None when the unwrap vote is off."""

    def __init__(self, cos_s, density, unwrapped, comp, density_scale, px_to_base, prominence, density_band):
        self.cos_s, self.density, self.unwrapped, self.comp = cos_s, density, unwrapped, comp
        self.density_scale, self.px_to_base, self.prominence = density_scale, px_to_base, prominence
        self.density_band = density_band

    def votes(self, a: np.ndarray, b: np.ndarray) -> dict:
        dens = density_integral(self.density, a, b, self.px_to_base) * self.density_scale
        v = {"density_raw": round(float(dens), 3), "density": int(np.rint(dens)),
             "ridges": interior_ridges(self.cos_s, a, b, self.prominence) + 1, "unwrap": None}
        if self.unwrapped is not None:
            ca, cb = self.comp[int(a[0]), int(a[1])], self.comp[int(b[0]), int(b[1])]
            if ca != 0 and ca == cb:
                va, vb = sample(self.unwrapped, np.array([a, b]))
                v["unwrap"] = int(np.rint((vb - va) / TWO_PI))
        return v

    def accept(self, v: dict, cert: str) -> bool:
        # A band on the raw integral, not rounding: a half-spacing hop integrates
        # to about 0.6 windings and rounding would let it through.
        dens_ok = abs(v["density_raw"] - 1.0) <= self.density_band
        if cert == "two":
            return dens_ok
        return dens_ok and v["unwrap"] == 1


def walk(seed: np.ndarray, nx: np.ndarray, ny: np.ndarray, valid: np.ndarray, cos_s: np.ndarray,
         certifier: Certifier, cert: str, rungs: int, max_len: float, prominence: float,
         axis_yx: tuple[float, float], min_radial: float) -> tuple[list, list]:
    """Chain hops from the seed outward; returns (points, hop votes).

    The outward sign of the normal comes from the umbilicus; where the sheet
    runs nearly radially (|n . r_hat| below min_radial) that sign is a coin
    toss, so the walk stops there.
    """
    pts, hops = [seed], []
    a = seed
    for _ in range(rungs):
        n = np.array([sample(ny, a[None, :])[0], sample(nx, a[None, :])[0]])  # (dy, dx)
        norm = np.hypot(*n)
        if norm < 0.3:
            break
        n /= norm
        r = a - np.array(axis_yx)
        if abs(float(np.dot(n, r)) / (np.linalg.norm(r) + 1e-9)) < min_radial:
            break
        b = next_ridge(cos_s, a, n, max_len, prominence)
        if b is None:
            break
        iy, ix = int(b[0]), int(b[1])
        if not (0 <= iy < valid.shape[0] - 1 and 0 <= ix < valid.shape[1] - 1) or not valid[iy, ix]:
            break
        v = certifier.votes(a, b)
        hops.append(v)
        if not certifier.accept(v, cert):
            break
        pts.append(b)
        a = b
    return pts, hops


def score_against_gold(ladders: list[np.ndarray], gold: list[np.ndarray], match: float, kmax: int) -> dict:
    """Gold pairs whose rungs both lie within `match` px of points of one of our ladders."""
    if not ladders:
        return {"tests": 0}
    pts = np.concatenate(ladders)
    owner = np.concatenate([np.full(len(l), i) for i, l in enumerate(ladders)])
    wind = np.concatenate([np.arange(len(l)) for l in ladders])
    tree = cKDTree(pts)
    per_k = {k: {"gold_pairs": 0, "tested": 0, "correct": 0, "correct_unsigned": 0} for k in range(1, kmax + 1)}
    failures = []
    for lad in gold:
        hits = [tree.query_ball_point(p, match) for p in lad]
        for k in range(1, kmax + 1):
            for i in range(len(lad) - k):
                per_k[k]["gold_pairs"] += 1
                la = {owner[h]: wind[h] for h in hits[i]}
                lb = {owner[h]: wind[h] for h in hits[i + k]}
                common = set(la) & set(lb)
                if not common:
                    continue
                per_k[k]["tested"] += 1
                if all(abs(lb[c] - la[c]) == k for c in common):
                    per_k[k]["correct_unsigned"] += 1
                if all(lb[c] - la[c] == k for c in common):
                    per_k[k]["correct"] += 1
                else:
                    failures.append({"k": k, "gold_a": [round(float(v), 1) for v in lad[i]], "gold_b": [round(float(v), 1) for v in lad[i + k]],
                                     "gold_rung_len": round(float(np.linalg.norm(lad[i + k] - lad[i])), 1),
                                     "ours": [{"ladder": int(c), "dwind": int(lb[c] - la[c]),
                                               "a": [round(float(v), 1) for v in ladders[c][la[c]]], "b": [round(float(v), 1) for v in ladders[c][lb[c]]],
                                               "dist_a": round(float(np.linalg.norm(ladders[c][la[c]] - lad[i])), 2),
                                               "dist_b": round(float(np.linalg.norm(ladders[c][lb[c]] - lad[i + k])), 2)} for c in common]})
    for k, r in per_k.items():
        r["coverage_of_gold"] = round(r["tested"] / r["gold_pairs"], 4) if r["gold_pairs"] else None
        r["precision"] = round(r["correct"] / r["tested"], 4) if r["tested"] else None
        r["precision_unsigned"] = round(r["correct_unsigned"] / r["tested"], 4) if r["tested"] else None
    tested = sum(r["tested"] for r in per_k.values())
    correct = sum(r["correct"] for r in per_k.values())
    unsigned = sum(r["correct_unsigned"] for r in per_k.values())
    return {"tests": tested, "correct": correct, "precision": round(correct / tested, 4) if tested else None,
            "correct_unsigned": unsigned, "precision_unsigned": round(unsigned / tested, 4) if tested else None,
            "per_k": per_k, "failures": failures}


def to_point_collections(ladders: list[np.ndarray], z0: int, y_off: int, x_off: int, cert: str) -> dict:
    # The fitter (villa spiral-fitting/point_collection.py) parses collection
    # keys as integers; a band file merges slices by offsetting the key.
    cols = {}
    for i, lad in enumerate(ladders):
        key = str(z0 * 100000 + i + 1)
        cols[key] = {"name": f"wp{z0}_{i + 1}",
                      "metadata": {"winding_is_absolute": False,
                                   "generator": f"winding-phase ladder (ridge walk, certified by {cert} votes)"},
                      "points": {str(j + 1): {"p": [2.0 * (p[1] + x_off), 2.0 * (p[0] + y_off), float(z0)], "wind_a": float(j + 1)}
                                 for j, p in enumerate(lad)}}
    return {"vc_pointcollections_json_version": "1", "collections": cols}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", default="PHerc0826", choices=sorted(SCROLLS))
    ap.add_argument("--z0", type=int, default=8528, help="level-0 z of the slice")
    ap.add_argument("--gold", type=Path, default=None)
    ap.add_argument("--cert", choices=["two", "three"], default="three")
    ap.add_argument("--stride", type=int, default=32, help="seed grid at level 1 (px)")
    ap.add_argument("--rungs", type=int, default=8)
    ap.add_argument("--max-len", type=float, default=30.0, help="longest hop (level-1 px)")
    ap.add_argument("--min-cos", type=float, default=0.6)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--smooth", type=float, default=0.3)
    ap.add_argument("--prominence", type=float, default=0.08)
    ap.add_argument("--density-scale", type=float, default=1.23)
    ap.add_argument("--density-band", type=float, default=0.25, help="accept a hop if |integral x scale - 1| <= band")
    ap.add_argument("--base-level", type=int, default=1)
    ap.add_argument("--match", type=float, default=2.5, help="gold rung to ladder point distance (level-1 px)")
    ap.add_argument("--min-radial", type=float, default=0.0, help="stop where |normal . radial| is below this (orientation unreliable)")
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    stores = SCROLLS[args.scroll]
    _, ny_, nx_ = level_shape(stores)
    z2 = args.z0 // 4
    if args.crop:
        h = args.crop // 2
        y0, y1, x0, x1 = ny_ // 2 - h, ny_ // 2 + h, nx_ // 2 - h, nx_ // 2 + h
    else:
        y0, y1, x0, x1 = 0, ny_, 0, nx_
    win = read_slice_window_cos_level1(stores, z2, y0, y1, x0, x1)
    axis = axis_at(load_control_points(DATA / f"{args.scroll}_umbilicus.json"), win.z, level=1)
    nx, ny, _ = orient_normals(win.nx, win.ny, win.y0, win.x0, axis)
    field = wrapped_phase(win.cos, nx, ny, window=args.window, smooth_sigma=args.smooth, grad_mag=win.grad_mag, min_grad_mag=1e-6)
    cos_s = gaussian_filter(win.cos, args.smooth)
    comp, _ = label(field.valid)
    unwrapped = None
    if args.cert == "three":
        masked = np.ma.array(field.phi.astype(np.float64), mask=~field.valid)
        unwrapped = np.nan_to_num(np.ma.filled(unwrap_phase(masked, rng=0), np.nan), nan=0.0)
    certifier = Certifier(cos_s, win.grad_mag, unwrapped, comp, args.density_scale, 2.0 ** (1 - args.base_level), args.prominence, args.density_band)

    seeds = grid_seeds(cos_s, ridge_mask(cos_s, nx, ny, args.min_cos) & field.valid, args.stride)
    ladders, hops = [], []
    for seed in seeds:
        pts, hv = walk(seed, nx, ny, field.valid, cos_s, certifier, args.cert, args.rungs, args.max_len, args.prominence, axis, args.min_radial)
        hops.extend(hv)
        if len(pts) >= 2:
            ladders.append(np.array(pts))

    n_hops = len(hops)
    stats = {"z0": args.z0, "crop": args.crop, "cert": args.cert, "density_band": args.density_band, "min_radial": args.min_radial,
             "seeds": int(len(seeds)), "ladders": len(ladders),
             "points": int(sum(len(l) for l in ladders)), "pairs_1wrap": int(sum(len(l) - 1 for l in ladders)),
             "hops_tried": n_hops,
             "hop_density_is_1": round(sum(h["density"] == 1 for h in hops) / n_hops, 4) if n_hops else None,
             "hop_unwrap_is_1": round(sum(h["unwrap"] == 1 for h in hops) / n_hops, 4) if n_hops else None,
             "hop_ridges_is_1": round(sum(h["ridges"] == 1 for h in hops) / n_hops, 4) if n_hops else None,
             "rung_len_px_p5_p50_p95": [round(float(v), 1) for v in np.percentile(
                 np.concatenate([np.linalg.norm(np.diff(l, axis=0), axis=1) for l in ladders]), [5, 50, 95])] if ladders else None,
             "ladder_len_hist": {int(k): int(v) for k, v in zip(*np.unique([len(l) for l in ladders], return_counts=True))} if ladders else {}}
    if args.gold:
        gold = [lad - np.array([win.y0, win.x0]) for lad in load_ladders(args.gold, args.z0, 0, target_level=1)]
        gold = [g for g in gold if ((g >= 0) & (g < np.array(cos_s.shape) - 1)).all()]
        stats["against_gold"] = score_against_gold(ladders, gold, args.match, args.kmax)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"z{args.z0}_{args.cert}_b{args.density_band:g}" + (f"_crop{args.crop}" if args.crop else "_full")
    (args.out / f"e03_ladders_{tag}.json").write_text(json.dumps(to_point_collections(ladders, args.z0, win.y0, win.x0, args.cert)))
    (args.out / f"e03_stats_{tag}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
