"""E-02a: baseline 2D unwrap of one PCU held-out slice, scored against PCU's gold ladders.

    python e02a_baseline.py --gold path/to/gold0826_heldout8.json --z0 8528 --out out/

Pipeline: read the level-1 slice (cos at level 1, normals upsampled), build
the wrapped phase (phase.py), unwrap it with scikit-image's reliability-
sorting algorithm (Herraez et al. 2002) on the valid mask, then for every
consecutive pair of points in every gold ladder on that slice, compare the
unwrapped phase difference / 2 pi (rounded) with the gold winding
difference (always +1 by construction of the ladders).

Reported: coverage (pairs with both points inside the same unwrapped
component), precision on covered pairs (rounded difference == +1), and the
distribution of rounded differences. No amplitude weighting and no
gradient prior: this is the floor, not the method.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label, map_coordinates
from skimage.restoration import unwrap_phase

from lasagna_store import PHERC0826_L2, read_slice_window_cos_level1
from phase import wrapped_phase
from umbilicus import axis_at, load_control_points, orient_normals

UMBILICUS = Path(__file__).parent / "data" / "PHerc0826_umbilicus.json"
TWO_PI = 2.0 * np.pi


def load_ladders(gold: Path, z0: int, xy_level: int, target_level: int) -> list[np.ndarray]:
    """Ladders on slice z0 (level-0 z) as (N, 2) arrays of (y, x) at target_level, ordered by wind_a."""
    d = json.loads(Path(gold).read_text())
    scale = 2.0 ** (xy_level - target_level)
    out = []
    for c in d["collections"].values():
        pts = sorted(c["points"].values(), key=lambda p: p["wind_a"])
        if round(pts[0]["p"][2]) != z0:
            continue
        out.append(np.array([(p["p"][1] * scale, p["p"][0] * scale) for p in pts], dtype=np.float64))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--z0", type=int, default=8528, help="level-0 z of the held-out slice")
    ap.add_argument("--xy-level", type=int, default=2, help="pyramid level of the gold x, y coordinates")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--smooth", type=float, default=0.7)
    ap.add_argument("--crop", type=int, default=None, help="centred square (level-2 px) for a quick run")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    z2 = args.z0 // 4
    if args.crop:
        c = 2043 // 2; h = args.crop // 2
        y0, y1, x0, x1 = c - h, c + h, c - h, c + h
    else:
        y0, y1, x0, x1 = 0, 2043, 0, 2043
    t0 = time.time()
    win = read_slice_window_cos_level1(PHERC0826_L2, z2, y0, y1, x0, x1)
    t_read = time.time() - t0
    axis = axis_at(load_control_points(UMBILICUS), win.z, level=1)
    nx, ny, _ = orient_normals(win.nx, win.ny, win.y0, win.x0, axis)
    field = wrapped_phase(win.cos, nx, ny, window=args.window, smooth_sigma=args.smooth, grad_mag=win.grad_mag, min_grad_mag=1e-6)

    t1 = time.time()
    masked = np.ma.array(field.phi.astype(np.float64), mask=~field.valid)
    unwrapped = np.ma.filled(unwrap_phase(masked, rng=0), np.nan)
    t_unwrap = time.time() - t1
    comp, ncomp = label(field.valid)

    ladders = load_ladders(args.gold, args.z0, args.xy_level, target_level=1)
    diffs, covered, total = [], 0, 0
    for lad in ladders:
        yx = lad - np.array([win.y0, win.x0])
        inside = (yx[:, 0] >= 0) & (yx[:, 0] < unwrapped.shape[0] - 1) & (yx[:, 1] >= 0) & (yx[:, 1] < unwrapped.shape[1] - 1)
        vals = np.full(len(lad), np.nan)
        comps = np.zeros(len(lad), dtype=int)
        if inside.any():
            vals[inside] = map_coordinates(np.nan_to_num(unwrapped, nan=0.0), yx[inside].T, order=1, mode="nearest")
            comps[inside] = comp[yx[inside, 0].astype(int), yx[inside, 1].astype(int)]
        for i in range(len(lad) - 1):
            if not (inside[i] and inside[i + 1]):
                continue  # outside the crop: not a test
            total += 1
            if comps[i] == 0 or comps[i] != comps[i + 1]:
                continue
            covered += 1
            diffs.append((vals[i + 1] - vals[i]) / TWO_PI)

    diffs = np.array(diffs)
    rounded = np.rint(diffs).astype(int) if diffs.size else np.array([], dtype=int)
    hist = {int(k): int(v) for k, v in zip(*np.unique(rounded, return_counts=True))} if diffs.size else {}
    correct = int((rounded == 1).sum())
    neg = int((rounded == -1).sum())
    res = {
        "z0": args.z0, "z_level1": win.z, "window": args.window, "crop": args.crop,
        "ladders_on_slice": len(ladders), "pairs_in_window": total, "pairs_covered": covered,
        "coverage": round(covered / total, 4) if total else None,
        "precision_plus1": round(correct / covered, 4) if covered else None,
        "precision_either_sign": round((correct + neg) / covered, 4) if covered else None,
        "rounded_diff_histogram": hist,
        "mean_abs_fractional_error": round(float(np.mean(np.abs(diffs - rounded))), 4) if diffs.size else None,
        "components": int(ncomp), "valid_fraction": round(float(field.valid.mean()), 4),
        "read_seconds": round(t_read, 1), "unwrap_seconds": round(t_unwrap, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"z{args.z0}" + (f"_crop{args.crop}" if args.crop else "_full") + (f"_w{args.window}_s{args.smooth:g}" if (args.window, args.smooth) != (30, 0.7) else "")
    (args.out / f"e02a_baseline_{tag}.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(args.out / f"e02a_unwrapped_{tag}.npz", unwrapped=unwrapped.astype(np.float32), valid=field.valid)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
