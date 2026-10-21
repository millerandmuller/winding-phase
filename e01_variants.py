"""E-01 variant comparison on one crop: what drives the residue count?

    python e01_variants.py --z 2250 --crop 640 --out out/

Variants:
  A  level 2, normals as stored
  B  level 2, normals oriented outward from the umbilicus
  C  level 1 cos (2x finer), normals oriented outward, window doubled

Reports residues per 1,000 valid pixels and the dipole fraction (a residue
with an opposite-sign residue within 2 px) for each.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from e01_residue_map import window_bounds
from lasagna_store import PHERC0826_L2, read_slice_window, read_slice_window_cos_level1
from phase import residues, wrapped_phase
from umbilicus import axis_at, load_control_points, orient_normals

UMBILICUS = Path(__file__).parent / "data" / "PHerc0826_umbilicus.json"


def summarise(name: str, field, res: np.ndarray) -> dict:
    pos, neg = res > 0, res < 0
    k = np.ones((5, 5), bool)
    paired = int((pos & binary_dilation(neg, k)).sum() + (neg & binary_dilation(pos, k)).sum())
    n = int(pos.sum() + neg.sum())
    nv = int(field.valid.sum())
    return {
        "variant": name,
        "valid_pixels": nv,
        "residues": n,
        "residues_per_1000_valid": round(1000.0 * n / nv, 2) if nv else None,
        "dipole_fraction": round(paired / n, 3) if n else None,
        "normal_agreement_rad_per_px": round(field.normal_agreement, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=int, default=2250)
    ap.add_argument("--crop", type=int, default=640)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    stores = PHERC0826_L2
    y0, y1, x0, x1 = window_bounds(args.crop)
    cps = load_control_points(UMBILICUS)
    rows = []

    win2 = read_slice_window(stores, args.z, y0, y1, x0, x1)
    f = wrapped_phase(win2.cos, win2.nx, win2.ny, window=args.window)
    rows.append(summarise("A_level2_stored_normals", f, residues(f)))

    axis2 = axis_at(cps, args.z, level=2)
    nx, ny, rdot = orient_normals(win2.nx, win2.ny, y0, x0, axis2)
    f = wrapped_phase(win2.cos, nx, ny, window=args.window)
    r = residues(f)
    rows.append(summarise("B_level2_oriented_normals", f, r))
    rows[-1]["umbilicus_level2_yx"] = [round(axis2[0], 1), round(axis2[1], 1)]
    rows[-1]["radial_dot_below_0.2_fraction"] = round(float((rdot[f.valid] < 0.2).mean()), 3)

    win1 = read_slice_window_cos_level1(stores, args.z, y0, y1, x0, x1)
    axis1 = axis_at(cps, win1.z, level=1)
    nx, ny, _ = orient_normals(win1.nx, win1.ny, win1.y0, win1.x0, axis1)
    f = wrapped_phase(win1.cos, nx, ny, window=2 * args.window)
    rows.append(summarise("C_level1_cos_oriented_normals", f, residues(f)))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"e01_variants_z{args.z}_crop{args.crop}.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
