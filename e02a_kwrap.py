"""E-02a: score the votes on 1-, 2- and 3-wrap pairs, not only on adjacent ones.

    python e02a_kwrap.py --gold gold0826_heldout8.json --z0 8528 --crop 800 --out out/

PCU's held-out ladders are chains of points on consecutive sheets. Points k
rungs apart are a true k-wrap pair, so the same slice yields negatives for
the +1 test. For k in 1, 2, 3 and for each estimate (2D unwrap, interior
ridge count + 1, integrated density) this reports accuracy per k, and for
each certificate (estimates that agree) the precision of the agreed value
and the coverage, over all pairs of all k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label, map_coordinates
from skimage.restoration import unwrap_phase

from e02a_baseline import load_ladders
from e02a_certify import density_integral, interior_ridges
from lasagna_store import PHERC0826_L2, read_slice_window_cos_level1
from phase import wrapped_phase
from umbilicus import axis_at, load_control_points, orient_normals

UMBILICUS = Path(__file__).parent / "data" / "PHerc0826_umbilicus.json"
TWO_PI = 2.0 * np.pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--z0", type=int, default=8528)
    ap.add_argument("--xy-level", type=int, default=0)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--smooth", type=float, default=0.3)
    ap.add_argument("--prominence", type=float, default=0.08)
    ap.add_argument("--base-level", type=int, default=1)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--density-scale", type=float, default=1.0, help="multiply the integrated density by this (calibrated on 1-wrap pairs: 1 / median integral)")
    ap.add_argument("--crop", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()
    px_to_base = 2.0 ** (1 - args.base_level)

    z2 = args.z0 // 4
    if args.crop:
        c = 2043 // 2; h = args.crop // 2
        y0, y1, x0, x1 = c - h, c + h, c - h, c + h
    else:
        y0, y1, x0, x1 = 0, 2043, 0, 2043
    win = read_slice_window_cos_level1(PHERC0826_L2, z2, y0, y1, x0, x1)
    axis = axis_at(load_control_points(UMBILICUS), win.z, level=1)
    nx, ny, _ = orient_normals(win.nx, win.ny, win.y0, win.x0, axis)
    field = wrapped_phase(win.cos, nx, ny, window=args.window, smooth_sigma=args.smooth, grad_mag=win.grad_mag, min_grad_mag=1e-6)
    masked = np.ma.array(field.phi.astype(np.float64), mask=~field.valid)
    unwrapped = np.nan_to_num(np.ma.filled(unwrap_phase(masked, rng=0), np.nan), nan=0.0)
    comp, _ = label(field.valid)
    cos_s = gaussian_filter(win.cos, args.smooth)

    rows = []
    for lad in load_ladders(args.gold, args.z0, args.xy_level, target_level=1):
        yx = lad - np.array([win.y0, win.x0])
        inside = (yx[:, 0] >= 0) & (yx[:, 0] < unwrapped.shape[0] - 1) & (yx[:, 1] >= 0) & (yx[:, 1] < unwrapped.shape[1] - 1)
        for k in range(1, args.kmax + 1):
            for i in range(len(lad) - k):
                if not (inside[i] and inside[i + k]):
                    continue
                a, b = yx[i], yx[i + k]
                ca, cb = comp[int(a[0]), int(a[1])], comp[int(b[0]), int(b[1])]
                row = {"k": k, "ridges": interior_ridges(cos_s, a, b, args.prominence) + 1,
                       "dens_raw": density_integral(win.grad_mag, a, b, px_to_base), "unwrap": None}
                row["dens"] = int(np.rint(row["dens_raw"] * args.density_scale))
                if ca != 0 and ca == cb:
                    va, vb = map_coordinates(unwrapped, np.array([a, b]).T, order=1, mode="nearest")
                    row["unwrap"] = int(np.rint((vb - va) / TWO_PI))
                rows.append(row)

    total = len(rows)
    per_k = {}
    for k in range(1, args.kmax + 1):
        rk = [r for r in rows if r["k"] == k]
        if not rk:
            continue
        unw = [r for r in rk if r["unwrap"] is not None]
        per_k[k] = {"pairs": len(rk),
                    "unwrap_acc": round(sum(r["unwrap"] == k for r in unw) / len(unw), 4) if unw else None,
                    "unwrap_covered": round(len(unw) / len(rk), 4),
                    "ridges_acc": round(sum(r["ridges"] == k for r in rk) / len(rk), 4),
                    "density_acc": round(sum(r["dens"] == k for r in rk) / len(rk), 4)}

    def cert(name, agree):
        sel = [r for r in rows if r["unwrap"] is not None and agree(r)]
        ok = sum(r["unwrap"] == r["k"] for r in sel)
        by_k = {k: {"accepted": sum(1 for r in sel if r["k"] == k), "correct": sum(1 for r in sel if r["k"] == k and r["unwrap"] == k)} for k in per_k}
        return {name: {"coverage": round(len(sel) / total, 4), "precision": round(ok / len(sel), 4) if sel else None, "n": len(sel), "by_k": by_k}}

    med1 = float(np.median([r["dens_raw"] for r in rows if r["k"] == 1])) if any(r["k"] == 1 for r in rows) else None
    result = {"z0": args.z0, "crop": args.crop, "pairs_total": total, "density_scale_used": args.density_scale,
              "median_raw_density_integral_k1": round(med1, 4) if med1 else None, "per_k": per_k}
    result.update(cert("unwrap+ridges", lambda r: r["unwrap"] == r["ridges"]))
    result.update(cert("unwrap+density", lambda r: r["unwrap"] == r["dens"]))
    result.update(cert("ridges+density (unwrap ignored for agreement, scored on unwrap)", lambda r: r["ridges"] == r["dens"]))
    result.update(cert("all three", lambda r: r["unwrap"] == r["ridges"] == r["dens"]))
    sel_rd = [r for r in rows if r["ridges"] == r["dens"]]
    result["ridges+density scored on ridges (no unwrap needed)"] = {
        "coverage": round(len(sel_rd) / total, 4), "precision": round(sum(r["ridges"] == r["k"] for r in sel_rd) / len(sel_rd), 4) if sel_rd else None, "n": len(sel_rd)}
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"z{args.z0}" + (f"_crop{args.crop}" if args.crop else "_full")
    (args.out / f"e02a_kwrap_{tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
