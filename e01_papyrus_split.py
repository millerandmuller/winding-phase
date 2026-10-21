"""E-01: residues on papyrus versus off papyrus, using the organizers' m7
surface prediction as the papyrus mask.

    python e01_papyrus_split.py out/e01_arrays_PHerc0826_L1_z4500_full_oriented.npz --z 2250 --out out/

The m7 store (surface-m7-L0-th0.2) is read at level 2 and upsampled to the
saved level-1 arrays. Pixels within `--near` px of a predicted surface count
as papyrus. The valid mask is eroded by `--erode` px first (border artifact).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, zoom

from lasagna_store import BUCKET, open_array

M7_URL = (
    f"{BUCKET}/PHerc0826/representations/predictions/surfaces/"
    "20250821151701-surface-20260413222639-surface-m7-L0-th0.2.zarr/2"
)


def dipole_mask(res: np.ndarray, radius: int = 2) -> np.ndarray:
    k = np.ones((2 * radius + 1, 2 * radius + 1), bool)
    pos, neg = res > 0, res < 0
    return (pos & binary_dilation(neg, k)) | (neg & binary_dilation(pos, k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--z", type=int, required=True, help="level-2 slice index")
    ap.add_argument("--threshold", type=int, default=128, help="m7 uint8 threshold for 'surface'")
    ap.add_argument("--near", type=int, default=3, help="dilate the surface by this many level-1 px")
    ap.add_argument("--erode", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    d = np.load(args.npz)
    valid, res = d["valid"], d["residues"]
    valid = binary_erosion(valid, np.ones((2 * args.erode + 1,) * 2, bool))
    corners = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    res = np.where(corners, res, 0).astype(np.int8)

    m7 = open_array(M7_URL)[args.z]
    m7_l1 = zoom(m7.astype(np.float32), (valid.shape[0] / m7.shape[0], valid.shape[1] / m7.shape[1]), order=1)
    surface = m7_l1 > args.threshold
    papyrus = binary_dilation(surface, np.ones((2 * args.near + 1,) * 2, bool))
    pap_cells = papyrus[:-1, :-1] & corners
    off_cells = ~papyrus[:-1, :-1] & corners
    dip = dipole_mask(res)

    def row(name: str, m: np.ndarray) -> dict:
        n, r, dp = int(m.sum()), int(((res != 0) & m).sum()), int((dip & m).sum())
        return {"region": name, "cells": n, "residues": r, "per_1000": round(1000.0 * r / n, 2) if n else None,
                "dipole_share": round(dp / r, 3) if r else None}

    out = {"m7": M7_URL, "threshold": args.threshold, "near_px": args.near, "erode_px": args.erode,
           "m7_surface_fraction_of_valid": round(float(surface[valid].mean()), 3),
           "rows": [row("papyrus (within near px of m7 surface)", pap_cells), row("off papyrus", off_cells),
                    row("all valid", corners)]}
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.npz.stem.replace("e01_arrays_", "")
    (args.out / f"e01_papyrus_split_{stem}.json").write_text(json.dumps(out, indent=2))
    np.save(args.out / f"e01_papyrus_mask_{stem}.npy", papyrus)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
