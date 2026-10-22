"""E-03: slips-per-wrap chart from the fitter eval lines.

    python e03_figure.py figures/e03/fits/eval_*.txt --out figures/e03/fits/slips_per_wrap.png

Each eval file holds one line of the form
    "<run>: slips S/T = f per wrap; unassigned points U"
written by PCU's exp/e43_eval_gold.py. Bars are ordered by the RUN_ORDER
below (unknown runs go last, alphabetically); the label carries S/T.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"^(?P<run>\S+): slips (?P<s>\d+)/(?P<t>\d+) = (?P<f>[\d.]+) per wrap; unassigned points (?P<u>\d+)")
RUN_ORDER = ["A0", "A0_s2", "PCU", "PCU_s2", "OURS", "OURS_s2", "OURS_w40", "OURS_d8400", "OURS_d8400_w40"]
LABELS = {"A0": "tracks only", "A0_s2": "tracks only, seed 2", "PCU": "PCU constraints", "PCU_s2": "PCU, seed 2",
          "OURS": "ours", "OURS_s2": "ours, seed 2", "OURS_w40": "ours, weight 40",
          "OURS_d8400": "ours, 8,400 draws", "OURS_d8400_w40": "ours, 8,400 draws, weight 40"}


def parse(files: list[Path]) -> dict:
    rows = {}
    for f in files:
        for line in f.read_text().splitlines():
            m = LINE.match(line.strip())
            if m:
                rows[m["run"]] = {"slips": int(m["s"]), "pairs": int(m["t"]), "rate": float(m["f"]), "unassigned": int(m["u"])}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = parse(args.files)
    order = [r for r in RUN_ORDER if r in rows] + sorted(r for r in rows if r not in RUN_ORDER)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(order) + 1.6), constrained_layout=True)
    colors = ["#888888" if r.startswith("A0") else "#4c72b0" if r.startswith("PCU") else "#dd8452" for r in order]
    ys = range(len(order))
    ax.barh(list(ys), [100 * rows[r]["rate"] for r in order], color=colors)
    for y, r in zip(ys, order):
        ax.text(100 * rows[r]["rate"] + 0.5, y, f"{100 * rows[r]['rate']:.1f}%  ({rows[r]['slips']:,} of {rows[r]['pairs']:,})", va="center", fontsize=9)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([LABELS.get(r, r) for r in order])
    ax.invert_yaxis()
    ax.set_xlim(0, max(100 * rows[r]["rate"] for r in order) + 18)
    ax.set_xlabel("slips per wrap on PCU's held-out ladders, PHerc0826 z 8500 to 9500 (lower is better)")
    ax.set_title("Spiral fit with different relative-winding constraints, same tracks, steps and seed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    for r in order:
        print(r, rows[r])


if __name__ == "__main__":
    main()
