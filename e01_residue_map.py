"""E-01: residue map of Lasagna's wrapped phase on one PHerc0826 slice, over the CT.

    python e01_residue_map.py --z 2250 --crop 640 --out out/

Reads one level-2 slice window from the open-data bucket, reconstructs the
wrapped phase (phase.py), computes residues, writes a three-panel PNG and a
JSON with the numbers. Nothing is written anywhere but --out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lasagna_store import PHERC0826_L2, SliceWindow, read_slice_window, read_slice_window_cos_level1
from umbilicus import axis_at, load_control_points, orient_normals

UMBILICUS = Path(__file__).parent / "data" / "PHerc0826_umbilicus.json"
from phase import PhaseField, residues, wrapped_phase

SLICE_SHAPE_L2 = (2043, 2043)


def window_bounds(crop: int | None) -> tuple[int, int, int, int]:
    """Full slice, or a centred square of side `crop`."""
    h, w = SLICE_SHAPE_L2
    if crop is None:
        return 0, h, 0, w
    cy, cx = h // 2, w // 2
    half = crop // 2
    return cy - half, cy + half, cx - half, cx + half


def stats(win: SliceWindow, field: PhaseField, res: np.ndarray) -> dict:
    n_valid = int(field.valid.sum())
    n_pos = int((res > 0).sum())
    n_neg = int((res < 0).sum())
    cells_valid = int((res != 0).size)
    return {
        "z_level2": win.z,
        "window_y0_x0": [win.y0, win.x0],
        "window_shape": list(win.cos.shape),
        "cos_zero_fraction": float((win.cos == 0).mean()),
        "valid_fraction": float(field.valid.mean()),
        "valid_pixels": n_valid,
        "residues_positive": n_pos,
        "residues_negative": n_neg,
        "residues_per_1000_valid_pixels": (1000.0 * (n_pos + n_neg) / n_valid) if n_valid else None,
        "normal_agreement_rad_per_px": field.normal_agreement,
        "contrast_median_in_valid": float(np.median(field.contrast[field.valid])) if n_valid else None,
        "cells_total": cells_valid,
    }


def render(win: SliceWindow, field: PhaseField, res: np.ndarray, png: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), constrained_layout=True)
    extent = (win.x0, win.x0 + win.cos.shape[1], win.y0 + win.cos.shape[0], win.y0)

    axes[0].imshow(win.ct, cmap="gray", extent=extent)
    axes[0].set_title(f"CT (level {"1 (upsampled from 2)" if win.z % 2 == 0 and win.cos.shape[0] > 2043 else 2})")

    phi_show = np.where(field.valid, field.phi, np.nan)
    axes[1].imshow(win.ct, cmap="gray", extent=extent, alpha=0.35)
    im = axes[1].imshow(phi_show, cmap="twilight", vmin=-np.pi, vmax=np.pi, extent=extent)
    axes[1].set_title("wrapped phase on the valid ribbon (grey = undefined)")
    fig.colorbar(im, ax=axes[1], shrink=0.7)

    axes[2].imshow(win.ct, cmap="gray", extent=extent)
    ys, xs = np.nonzero(res > 0)
    axes[2].scatter(xs + win.x0 + 0.5, ys + win.y0 + 0.5, s=4, c="red", label=f"+1 ({len(xs)})")
    ys, xs = np.nonzero(res < 0)
    axes[2].scatter(xs + win.x0 + 0.5, ys + win.y0 + 0.5, s=4, c="deepskyblue", label=f"-1 ({len(xs)})")
    axes[2].legend(loc="upper right")
    axes[2].set_title("residues over CT")

    fig.suptitle(title)
    fig.savefig(png, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--z", type=int, default=2250, help="level-2 z index (band 8500-9492 is 2125-2373)")
    ap.add_argument("--crop", type=int, default=None, help="centred square side in level-2 pixels; omit for the full slice")
    ap.add_argument("--window", type=int, default=15, help="contrast window in level-2 pixels (doubled at level 1)")
    ap.add_argument("--level1", action="store_true", help="read cos at pyramid level 1 (2x finer); normals and CT upsampled")
    ap.add_argument("--smooth", type=float, default=0.7, help="gaussian sigma on cos before normalisation (px at the working level)")
    ap.add_argument("--min-grad-mag", type=float, default=0.0, help="drop pixels whose Lasagna grad_mag is below this (decoded units)")
    ap.add_argument("--orient", action="store_true", help="orient normals outward from the umbilicus (data/PHerc0826_umbilicus.json)")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    stores = PHERC0826_L2
    y0, y1, x0, x1 = window_bounds(args.crop)
    t0 = time.time()
    if args.level1:
        win = read_slice_window_cos_level1(stores, args.z, y0, y1, x0, x1)
        window, level = 2 * args.window, 1
    else:
        win = read_slice_window(stores, args.z, y0, y1, x0, x1)
        window, level = args.window, 2
    t_read = time.time() - t0

    nx, ny = win.nx, win.ny
    if args.orient:
        axis = axis_at(load_control_points(UMBILICUS), win.z, level=level)
        nx, ny, _ = orient_normals(nx, ny, win.y0, win.x0, axis)
    field = wrapped_phase(win.cos, nx, ny, window=window, grad_mag=win.grad_mag, min_grad_mag=args.min_grad_mag, smooth_sigma=args.smooth)
    res = residues(field)
    s = stats(win, field, res)
    s["read_seconds"] = round(t_read, 1)
    s["window"] = window
    s["level"] = level
    s["normals_oriented"] = bool(args.orient)
    s["min_grad_mag"] = args.min_grad_mag
    s["source"] = {"lasagna": stores.lasagna_root, "ct": stores.ct_url}

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"PHerc0826_L{level}_z{win.z}_" + ("full" if args.crop is None else f"crop{args.crop}") + ("_oriented" if args.orient else "") + (f"_gm{args.min_grad_mag:g}" if args.min_grad_mag else "") + (f"_s{args.smooth:g}" if args.smooth != 0.7 else "")
    png = args.out / f"e01_residues_{tag}.png"
    render(win, field, res, png, f"E-01 {tag}: residues of Lasagna wrapped phase")
    (args.out / f"e01_stats_{tag}.json").write_text(json.dumps(s, indent=2))
    np.savez_compressed(args.out / f"e01_arrays_{tag}.npz", phi=field.phi, valid=field.valid, residues=res)
    print(json.dumps(s, indent=2))
    print("wrote", png)


if __name__ == "__main__":
    main()
