"""Zoomed windows of a saved E-01 result at native pixels, residues over the CT.

    python e01_zoom.py out/e01_arrays_PHerc0826_L1_z4500_full_oriented.npz --z 2250 \
        --window core --window outer --size 400 --out out/

Named windows are centred on the umbilicus (`core`), on the point of highest
residue density (`densest`), or on explicit `y,x` level-1 coordinates. The CT
is re-read for the window only (level 2, upsampled), the phase and residues
come from the saved arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter, zoom

from lasagna_store import PHERC0826_L2, open_array
from umbilicus import axis_at, load_control_points

UMBILICUS = Path(__file__).parent / "data" / "PHerc0826_umbilicus.json"


def read_ct_level1(z_l1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    ct = open_array(PHERC0826_L2.ct_url)[z_l1 // 2, y0 // 2 : (y1 + 1) // 2, x0 // 2 : (x1 + 1) // 2]
    up = zoom(ct.astype(np.float32), 2, order=1)
    return up[: y1 - y0, : x1 - x0]


def centre_for(name: str, res: np.ndarray, z_l2: int) -> tuple[int, int]:
    if name == "core":
        y, x = axis_at(load_control_points(UMBILICUS), 2 * z_l2, level=1)  # level-1 slice index
        return int(y), int(x)
    if name == "densest":
        dens = uniform_filter((res != 0).astype(np.float32), size=201)
        y, x = np.unravel_index(int(np.argmax(dens)), dens.shape)
        return int(y), int(x)
    y, x = (int(v) for v in name.split(","))
    return y, x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--z", type=int, required=True, help="level-2 slice index of the saved result")
    ap.add_argument("--window", action="append", default=[], help="core | densest | y,x (level-1 px); repeatable")
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    d = np.load(args.npz)
    phi, valid, res = d["phi"], d["valid"], d["residues"]
    half = args.size // 2
    stem = args.npz.stem.replace("e01_arrays_", "")
    for name in args.window or ["core"]:
        cy, cx = centre_for(name, res, args.z)
        y0, y1 = max(0, cy - half), min(phi.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(phi.shape[1], cx + half)
        ct = read_ct_level1(2 * args.z, y0, y1, x0, x1)
        extent = (x0, x1, y1, y0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
        phi_show = np.where(valid[y0:y1, x0:x1], phi[y0:y1, x0:x1], np.nan)
        axes[0].imshow(ct, cmap="gray", extent=extent, alpha=0.4)
        axes[0].imshow(phi_show, cmap="twilight", vmin=-np.pi, vmax=np.pi, extent=extent, alpha=0.8)
        axes[0].set_title("wrapped phase over CT")
        axes[1].imshow(ct, cmap="gray", extent=extent)
        r = res[y0:y1, x0:x1]
        ys, xs = np.nonzero(r > 0)
        axes[1].scatter(xs + x0 + 0.5, ys + y0 + 0.5, s=14, c="red", marker="+", label=f"+1 ({len(xs)})")
        ys, xs = np.nonzero(r < 0)
        axes[1].scatter(xs + x0 + 0.5, ys + y0 + 0.5, s=14, c="deepskyblue", marker="x", label=f"-1 ({len(xs)})")
        axes[1].legend(loc="upper right")
        axes[1].set_title("residues over CT")
        label = name.replace(",", "_")
        fig.suptitle(f"E-01 zoom {stem}: {name} at level-1 ({cy}, {cx}), {args.size} px")
        png = args.out / f"e01_zoom_{stem}_{label}.png"
        fig.savefig(png, dpi=110)
        plt.close(fig)
        print("wrote", png, "centre", (cy, cx), "residues in window", int((r != 0).sum()))


if __name__ == "__main__":
    main()
