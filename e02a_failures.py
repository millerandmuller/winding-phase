"""E-02a follow-up: where do the baseline's wrong pairs sit?

    python e02a_failures.py --gold gold0826_heldout8.json --z0 8528 --crop 800 --out out/

Draws every scored gold pair over the wrapped phase: green = +1 (right),
orange = 0 (missed wrap), blue = -1 (reversed), red = anything else, grey =
not covered. Tabulates the failure rate by local fringe period (from the
unwrapped phase gradient along the pair) and by pair length.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, map_coordinates
from skimage.restoration import unwrap_phase

from e02a_baseline import load_ladders
from lasagna_store import PHERC0826_L2, read_slice_window_cos_level1
from phase import wrapped_phase
from umbilicus import axis_at, load_control_points, orient_normals

UMBILICUS = Path(__file__).parent / "data" / "PHerc0826_umbilicus.json"
TWO_PI = 2.0 * np.pi
COLOURS = {1: "lime", 0: "orange", -1: "deepskyblue"}


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
    masked = np.ma.array(field.phi.astype(np.float64), mask=~field.valid)
    unwrapped = np.nan_to_num(np.ma.filled(unwrap_phase(masked, rng=0), np.nan), nan=0.0)
    comp, _ = label(field.valid)

    pairs = []
    for lad in load_ladders(args.gold, args.z0, args.xy_level, target_level=1):
        yx = lad - np.array([win.y0, win.x0])
        inside = (yx[:, 0] >= 0) & (yx[:, 0] < unwrapped.shape[0] - 1) & (yx[:, 1] >= 0) & (yx[:, 1] < unwrapped.shape[1] - 1)
        for i in range(len(lad) - 1):
            if not (inside[i] and inside[i + 1]):
                continue
            a, b = yx[i], yx[i + 1]
            ca, cb = comp[int(a[0]), int(a[1])], comp[int(b[0]), int(b[1])]
            if ca == 0 or ca != cb:
                pairs.append((a, b, None)); continue
            va, vb = map_coordinates(unwrapped, np.array([a, b]).T, order=1, mode="nearest")
            pairs.append((a, b, int(np.rint((vb - va) / TWO_PI))))

    # failure rate by pair length (level-1 px) and by radius from the umbilicus (mm)
    ay, ax_ = axis[0] - win.y0, axis[1] - win.x0
    rows = []
    for a, b, k in pairs:
        if k is None:
            continue
        L = float(np.hypot(*(b - a)))
        r_mm = float(np.hypot(*((a + b) / 2 - np.array([ay, ax_]))) * 2 * 9.362 / 1000.0)
        rows.append((L, r_mm, k))
    rows = np.array(rows)

    def table(values, edges, name):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (values >= lo) & (values < hi)
            n = int(m.sum())
            if n == 0:
                continue
            ks = rows[m, 2]
            out.append({name: f"{lo}-{hi}", "n": n, "plus1": round(float((ks == 1).mean()), 3),
                        "zero": round(float((ks == 0).mean()), 3), "minus1": round(float((ks == -1).mean()), 3),
                        "other": round(float((np.abs(ks) > 1).mean() + 0.0 - 0.0 if True else 0), 3)})
        return out

    result = {
        "pairs_scored": int(len(rows)), "not_covered": int(sum(1 for p in pairs if p[2] is None)),
        "by_pair_length_px": table(rows[:, 0], [0, 8, 10, 12, 14, 16, 20, 30, 1000], "length_px"),
        "by_radius_mm": table(rows[:, 1], [0, 2, 4, 6, 8, 10, 15, 100], "radius_mm"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"z{args.z0}" + (f"_crop{args.crop}" if args.crop else "_full")
    (args.out / f"e02a_failures_{tag}.json").write_text(json.dumps(result, indent=2))

    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    extent = (win.x0, win.x0 + field.phi.shape[1], win.y0 + field.phi.shape[0], win.y0)
    ax.imshow(np.where(field.valid, field.phi, np.nan), cmap="twilight", vmin=-np.pi, vmax=np.pi, extent=extent, alpha=0.6)
    for a, b, k in pairs:
        col = "grey" if k is None else COLOURS.get(k, "red")
        ax.plot([a[1] + win.x0, b[1] + win.x0], [a[0] + win.y0, b[0] + win.y0], color=col, lw=1.6)
    ax.set_title(f"E-02a {tag}: gold pairs, green +1, orange 0, blue -1, red other, grey uncovered")
    fig.savefig(args.out / f"e02a_failures_{tag}.png", dpi=110)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
