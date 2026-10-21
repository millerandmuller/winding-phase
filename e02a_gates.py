"""E-02a follow-up: can a confidence gate turn the baseline into PCU's shape?

    python e02a_gates.py --gold gold0826_heldout8.json --z0 8528 --crop 800 --out out/

Recomputes the phase and the baseline unwrap on the window, then scores each
gold pair with three candidate confidences and sweeps thresholds on each,
reporting precision (rounded difference == +1) against coverage:

  frac   : |d - round(d)| of the unwrapped phase difference d (in windings)
  amp    : minimum local cosine amplitude (contrast) sampled along the segment
  resdist: distance from the segment's midpoint to the nearest residue (px)
  combo  : frac below 0.15 and amp above the amp threshold

The target shape to beat is PCU's 99.6% precision at 70% coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label, map_coordinates
from skimage.restoration import unwrap_phase

from e02a_baseline import load_ladders
from lasagna_store import PHERC0826_L2, read_slice_window_cos_level1
from phase import residues, wrapped_phase
from umbilicus import axis_at, load_control_points, orient_normals

UMBILICUS = Path(__file__).parent / "data" / "PHerc0826_umbilicus.json"
TWO_PI = 2.0 * np.pi


def segment_min(arr: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 9) -> float:
    t = np.linspace(0.0, 1.0, n)
    pts = a[None, :] * (1 - t)[:, None] + b[None, :] * t[:, None]
    return float(np.min(map_coordinates(arr, pts.T, order=1, mode="nearest")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--z0", type=int, default=8528)
    ap.add_argument("--xy-level", type=int, default=0)
    ap.add_argument("--window", type=int, default=30)
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
    field = wrapped_phase(win.cos, nx, ny, window=args.window)
    res = residues(field)
    masked = np.ma.array(field.phi.astype(np.float64), mask=~field.valid)
    unwrapped = np.nan_to_num(np.ma.filled(unwrap_phase(masked, rng=0), np.nan), nan=0.0)
    comp, _ = label(field.valid)
    res_full = np.zeros(field.phi.shape, bool)
    res_full[:-1, :-1] = res != 0
    resdist = distance_transform_edt(~res_full)

    rows = []
    for lad in load_ladders(args.gold, args.z0, args.xy_level, target_level=1):
        yx = lad - np.array([win.y0, win.x0])
        inside = (yx[:, 0] >= 0) & (yx[:, 0] < unwrapped.shape[0] - 1) & (yx[:, 1] >= 0) & (yx[:, 1] < unwrapped.shape[1] - 1)
        for i in range(len(lad) - 1):
            if not (inside[i] and inside[i + 1]):
                continue
            a, b = yx[i], yx[i + 1]
            ca, cb = comp[int(a[0]), int(a[1])], comp[int(b[0]), int(b[1])]
            if ca == 0 or ca != cb:
                rows.append({"covered": False})
                continue
            va, vb = map_coordinates(unwrapped, np.array([a, b]).T, order=1, mode="nearest")
            d = (vb - va) / TWO_PI
            mid = (a + b) / 2
            rows.append({
                "covered": True,
                "d": float(d),
                "ok": int(np.rint(d)) == 1,
                "frac": float(abs(d - np.rint(d))),
                "amp": segment_min(field.contrast, a, b),
                "resdist": float(map_coordinates(resdist, mid[None, :].T, order=1, mode="nearest")[0]),
            })

    total = len(rows)
    cov = [r for r in rows if r["covered"]]
    base = {"pairs": total, "covered": len(cov), "coverage": round(len(cov) / total, 4),
            "precision": round(sum(r["ok"] for r in cov) / len(cov), 4)}

    def sweep(key: str, thresholds: list[float], keep_above: bool) -> list[dict]:
        out = []
        for t in thresholds:
            sel = [r for r in cov if (r[key] >= t if keep_above else r[key] <= t)]
            if not sel:
                out.append({"threshold": t, "coverage": 0.0, "precision": None}); continue
            out.append({"threshold": t, "coverage": round(len(sel) / total, 4),
                        "precision": round(sum(r["ok"] for r in sel) / len(sel), 4), "n": len(sel)})
        return out

    result = {
        "z0": args.z0, "crop": args.crop, "baseline": base,
        "gate_frac_max": sweep("frac", [0.5, 0.3, 0.2, 0.15, 0.1, 0.05], keep_above=False),
        "gate_amp_min": sweep("amp", [0.0, 0.2, 0.3, 0.4, 0.5, 0.6], keep_above=True),
        "gate_resdist_min": sweep("resdist", [0.0, 2.0, 4.0, 8.0, 12.0], keep_above=True),
    }
    combo = []
    for ta in (0.2, 0.3, 0.4, 0.5):
        sel = [r for r in cov if r["frac"] <= 0.15 and r["amp"] >= ta]
        combo.append({"frac_max": 0.15, "amp_min": ta, "coverage": round(len(sel) / total, 4),
                      "precision": round(sum(r["ok"] for r in sel) / len(sel), 4) if sel else None, "n": len(sel)})
    result["gate_combo"] = combo
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"z{args.z0}" + (f"_crop{args.crop}" if args.crop else "_full")
    (args.out / f"e02a_gates_{tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
