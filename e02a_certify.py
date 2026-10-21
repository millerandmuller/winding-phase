"""E-02a follow-up: a local certificate for each gold pair, and its agreement
with the global unwrap.

    python e02a_certify.py --gold gold0826_heldout8.json --z0 8528 --crop 800 --out out/

For each consecutive gold pair (two points on adjacent sheets), sample the
smoothed cosine along the segment and count interior ridges (local maxima
above a prominence). Adjacent sheets should show no interior ridge; each
interior ridge means a skipped sheet. The ridge count plus one is a local
winding estimate; the unwrap gives a global one. Reports precision and
coverage for: unwrap alone, ridge count alone, and pairs where both agree
(the certificate), against the gold answer of +1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label, map_coordinates
from scipy.signal import find_peaks
from skimage.restoration import unwrap_phase

from e02a_baseline import load_ladders
from lasagna_store import PHERC0826_L2, read_slice_window_cos_level1
from phase import wrapped_phase
from umbilicus import axis_at, load_control_points, orient_normals

UMBILICUS = Path(__file__).parent / "data" / "PHerc0826_umbilicus.json"
TWO_PI = 2.0 * np.pi


def interior_ridges(cos_s: np.ndarray, a: np.ndarray, b: np.ndarray, prominence: float, step: float = 0.5) -> int:
    n = max(5, int(np.hypot(*(b - a)) / step))
    t = np.linspace(0.0, 1.0, n)
    pts = a[None, :] * (1 - t)[:, None] + b[None, :] * t[:, None]
    prof = map_coordinates(cos_s, pts.T, order=1, mode="nearest")
    peaks, _ = find_peaks(prof, prominence=prominence)
    margin = max(1, int(0.15 * n))  # endpoints sit on ridges; ignore peaks near them
    return int(np.sum((peaks > margin) & (peaks < n - 1 - margin)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--z0", type=int, default=8528)
    ap.add_argument("--xy-level", type=int, default=0)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--smooth", type=float, default=0.7)
    ap.add_argument("--prominence", type=float, default=0.05)
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    z2 = args.z0 // 4
    if args.crop:
        c = 2043 // 2; h = args.crop // 2
        y0, y1, x0, x1 = c - h, c + h, c - h, c + h
    else:
        y0, y1, x0, x1 = 0, 2043, 0, 2043
    win = read_slice_window_cos_level1(PHERC0826_L2, z2, y0, y1, x0, x1)
    axis = axis_at(load_control_points(UMBILICUS), win.z, level=1)
    nx, ny, _ = orient_normals(win.nx, win.ny, win.y0, win.x0, axis)
    field = wrapped_phase(win.cos, nx, ny, window=args.window, smooth_sigma=args.smooth)
    masked = np.ma.array(field.phi.astype(np.float64), mask=~field.valid)
    unwrapped = np.nan_to_num(np.ma.filled(unwrap_phase(masked, rng=0), np.nan), nan=0.0)
    comp, _ = label(field.valid)
    cos_s = gaussian_filter(win.cos, args.smooth)

    rows = []
    for lad in load_ladders(args.gold, args.z0, args.xy_level, target_level=1):
        yx = lad - np.array([win.y0, win.x0])
        inside = (yx[:, 0] >= 0) & (yx[:, 0] < unwrapped.shape[0] - 1) & (yx[:, 1] >= 0) & (yx[:, 1] < unwrapped.shape[1] - 1)
        for i in range(len(lad) - 1):
            if not (inside[i] and inside[i + 1]):
                continue
            a, b = yx[i], yx[i + 1]
            ca, cb = comp[int(a[0]), int(a[1])], comp[int(b[0]), int(b[1])]
            ridges = interior_ridges(cos_s, a, b, args.prominence)
            if ca == 0 or ca != cb:
                rows.append({"unwrap": None, "ridges": ridges}); continue
            va, vb = map_coordinates(unwrapped, np.array([a, b]).T, order=1, mode="nearest")
            rows.append({"unwrap": int(np.rint((vb - va) / TWO_PI)), "ridges": ridges})

    total = len(rows)

    def pc(sel, pred_ok):
        n = len(sel)
        return {"coverage": round(n / total, 4), "precision": round(sum(pred_ok(r) for r in sel) / n, 4) if n else None, "n": n}

    unw = [r for r in rows if r["unwrap"] is not None]
    result = {
        "z0": args.z0, "crop": args.crop, "pairs": total, "prominence": args.prominence,
        "unwrap_alone": pc(unw, lambda r: r["unwrap"] == 1),
        "ridges_alone (predict ridges+1 == 1)": pc(rows, lambda r: r["ridges"] == 0),
        "ridge_count_histogram": {int(k): int(v) for k, v in zip(*np.unique([r["ridges"] for r in rows], return_counts=True))},
        "certificate (unwrap == ridges+1 == 1)": pc([r for r in unw if r["unwrap"] == r["ridges"] + 1], lambda r: r["unwrap"] == 1),
        "agree_any_value": pc([r for r in unw if r["unwrap"] == r["ridges"] + 1], lambda r: True),
        "unwrap_1_but_ridges_not_0": sum(1 for r in unw if r["unwrap"] == 1 and r["ridges"] != 0),
        "ridges_0_but_unwrap_not_1": sum(1 for r in unw if r["ridges"] == 0 and r["unwrap"] != 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"z{args.z0}" + (f"_crop{args.crop}" if args.crop else "_full")
    (args.out / f"e02a_certify_{tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
