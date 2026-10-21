"""E-01 follow-up: are the residues noise or topology?

    python e01_residue_analysis.py out/e01_arrays_PHerc0826_L1_z4500_full_oriented.npz --z 2250 --out out/

Two tests on a saved full-slice result:

1. Residue density against the local fringe period. Period is estimated from
   the wrapped phase itself (2 pi / |grad phi| along the normal, median in a
   window). If density rises steeply as the period approaches the sampling
   limit, the residues there are sampling noise.
2. Dipole share by radial band from the umbilicus (inner core, mid, outer
   wraps). Dipoles (a residue with an opposite-sign residue within 2 px) are
   the noise signature; extended same-sign lines are topology.

Writes a JSON table and a PNG with the density curve. Needs the level-2
normals and grad_mag again for the period estimate, so it re-reads one slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, median_filter

from phase import wrap
from umbilicus import axis_at, load_control_points

UMBILICUS = Path(__file__).parent / "data" / "PHerc0826_umbilicus.json"


def local_period(phi: np.ndarray, valid: np.ndarray, window: int = 15) -> np.ndarray:
    """Fringe period in pixels from the wrapped phase gradient magnitude."""
    gx = np.zeros_like(phi)
    gy = np.zeros_like(phi)
    gx[:, :-1] = wrap(phi[:, 1:] - phi[:, :-1])
    gy[:-1, :] = wrap(phi[1:, :] - phi[:-1, :])
    g = np.hypot(gx, gy)
    g = median_filter(np.where(valid, g, np.nan), size=window, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(g > 0, 2.0 * np.pi / g, np.nan)


def dipole_mask(res: np.ndarray, radius: int = 2) -> np.ndarray:
    k = np.ones((2 * radius + 1, 2 * radius + 1), bool)
    pos, neg = res > 0, res < 0
    return (pos & binary_dilation(neg, k)) | (neg & binary_dilation(pos, k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--z", type=int, required=True, help="level-2 slice index of the saved result")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--erode", type=int, default=0, help="erode the valid mask by this many pixels before counting (border-artifact test)")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    d = np.load(args.npz)
    phi, valid, res = d["phi"], d["valid"], d["residues"]
    if args.erode > 0:
        from scipy.ndimage import binary_erosion
        valid = binary_erosion(valid, np.ones((2 * args.erode + 1,) * 2, bool))
        corners = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
        res = np.where(corners, res, 0).astype(np.int8)
    period = local_period(phi, valid)
    has_res = res != 0
    dip = dipole_mask(res)

    # 1. density by period bin (cells share the pixel grid up to one row/col)
    p_cells = period[:-1, :-1]
    v_cells = valid[:-1, :-1] & np.isfinite(p_cells)
    edges = np.array([0, 3, 4, 5, 6, 8, 10, 14, 20, 40, 1e9])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = v_cells & (p_cells >= lo) & (p_cells < hi)
        n = int(m.sum())
        r = int((has_res & m).sum())
        dp = int((dip & m).sum())
        rows.append({"period_px_lo": float(lo), "period_px_hi": float(min(hi, 1e6)), "cells": n,
                     "residues": r, "per_1000": round(1000.0 * r / n, 2) if n else None,
                     "dipole_share": round(dp / r, 3) if r else None})

    # 2. dipole share by radial band
    scale = 2 ** args.level
    z_level = args.z * (2 ** (2 - args.level))
    ay, ax = axis_at(load_control_points(UMBILICUS), z_level, level=args.level)
    yy, xx = np.mgrid[0 : res.shape[0], 0 : res.shape[1]]
    rad = np.hypot(yy - ay, xx - ax) * scale * 9.362 / 1000.0  # mm: one level-0 voxel is 9.362 um
    bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 30), (30, 1e9)]
    radial = []
    for lo, hi in bands:
        m = (rad >= lo) & (rad < hi) & valid[:-1, :-1]
        n, r, dp = int(m.sum()), int((has_res & m).sum()), int((dip & m).sum())
        radial.append({"radius_mm_lo": lo, "radius_mm_hi": min(hi, 1e6), "cells": n, "residues": r,
                       "per_1000": round(1000.0 * r / n, 2) if n else None, "dipole_share": round(dp / r, 3) if r else None})

    out = {"npz": str(args.npz), "umbilicus_level_yx": [round(ay, 1), round(ax, 1)],
           "total_residues": int(has_res.sum()), "total_dipole_share": round(float(dip.sum() / max(1, has_res.sum())), 3),
           "by_period": rows, "by_radius_mm": radial}
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.npz.stem.replace("e01_arrays_", "") + (f"_erode{args.erode}" if args.erode else "")
    (args.out / f"e01_analysis_{stem}.json").write_text(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    xs = [f"{r['period_px_lo']:.0f}-{r['period_px_hi']:.0f}" if r['period_px_hi'] < 1e6 else f">{r['period_px_lo']:.0f}" for r in rows]
    axes[0].bar(xs, [r["per_1000"] or 0 for r in rows])
    axes[0].set_xlabel("local fringe period (px)")
    axes[0].set_ylabel("residues per 1,000 cells")
    axes[0].set_title("residue density vs fringe period")
    axes[1].bar([f"{b['radius_mm_lo']}-{b['radius_mm_hi']}" if b['radius_mm_hi'] < 1e6 else f">{b['radius_mm_lo']}" for b in radial],
                [b["dipole_share"] or 0 for b in radial])
    axes[1].set_xlabel("distance from umbilicus (mm)")
    axes[1].set_ylabel("dipole share of residues")
    axes[1].set_title("noise signature by radius")
    fig.suptitle(f"E-01 analysis {stem}")
    fig.savefig(args.out / f"e01_analysis_{stem}.png", dpi=110)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
