"""Correct pairs out of all pairs, next to PCU's slips per wrap.

    python e03_correct_of_all.py --pcu path/to/vesuvius-scrolling human A0=a.npz OURS=b.npz --ref A0 --out out.json
    LADDERS=gold.json Z0=9500 Z1=10500 python e03_correct_of_all.py --pcu ... gold A0=a.npz ... --ref A0 --out out.json

Slips per wrap divides by the pairs a fit assigns; a pair whose point lies more than
10 voxels from the fitted surface drops out of both counts, so a fit can lower its
rate by assigning fewer pairs. This script runs PCU's own exp/e31b_bootstrap.py
unchanged (same ladders, same scoring), takes its per-ladder slip and assigned-pair
counts, and adds per fit:

  correct_of_all = (assigned pairs - slips) / all pairs   (unassigned counts as wrong)

with a 95% ladder-level bootstrap interval and paired differences against each --ref run,
using the same resampling as PCU's script (2,000 resamples, generator seed 1).
The per-ladder counts are written too, so pooled results across bands can be computed.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

import numpy as np


def run_pcu_scoring(pcu: Path, kind: str, fits: list[str], ref: str | None, pcu_out: Path) -> dict:
    """Run PCU's e31b_bootstrap.py in-process and return its module globals."""
    script = pcu / "exp" / "e31b_bootstrap.py"
    argv = [str(script), kind, *fits, "--out", str(pcu_out)] + (["--ref", ref] if ref else [])
    old_argv, old_path = sys.argv, list(sys.path)
    sys.argv = argv
    sys.path.insert(0, str(pcu))
    try:
        return runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv, sys.path[:] = old_argv, old_path


def correct_of_all(S: np.ndarray, T: np.ndarray, N: np.ndarray, idx: np.ndarray) -> float:
    return float((T[idx] - S[idx]).sum() / max(N[idx].sum(), 1))


def summarize(res: dict, pairs_per_ladder: np.ndarray, refs: list[str], nboot: int) -> dict:
    n = len(pairs_per_ladder)
    boot = np.random.default_rng(1).integers(0, n, (nboot, n))
    every = np.arange(n)
    arrays = {k: (np.array(v["S"]), np.array(v["T"])) for k, v in res.items()}
    out = {}
    for name, (S, T) in arrays.items():
        vals = np.array([correct_of_all(S, T, pairs_per_ladder, b) for b in boot])
        row = {"correct": int((T - S).sum()), "all_pairs": int(pairs_per_ladder.sum()),
               "correct_of_all": correct_of_all(S, T, pairs_per_ladder, every),
               "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
               "slips": int(S.sum()), "assigned_pairs": int(T.sum()), "slips_per_wrap": float(S.sum() / max(T.sum(), 1)),
               "per_ladder_slips": S.tolist(), "per_ladder_assigned": T.tolist()}
        row["diff_vs"] = {}
        for ref in refs:
            if ref == name:
                continue
            Sr, Tr = arrays[ref]
            d = np.array([correct_of_all(S, T, pairs_per_ladder, b) - correct_of_all(Sr, Tr, pairs_per_ladder, b) for b in boot])
            row["diff_vs"][ref] = {"mean": row["correct_of_all"] - correct_of_all(Sr, Tr, pairs_per_ladder, every),
                                   "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        out[name] = row
    out["per_ladder_pairs"] = pairs_per_ladder.tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcu", type=Path, required=True, help="clone of github.com/Jashann/vesuvius-scrolling")
    ap.add_argument("kind", choices=["human", "gold"])
    ap.add_argument("fits", nargs="+", help="NAME=grids.npz")
    ap.add_argument("--ref", default=None, help="comma list of reference runs; PCU's script gets the first")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    refs = args.ref.split(",") if args.ref else []
    missing = set(refs) - {f.split("=")[0] for f in args.fits}
    if missing:
        raise SystemExit(f"--ref {sorted(missing)} not among the fits")
    g = run_pcu_scoring(args.pcu.resolve(), args.kind, args.fits, refs[0] if refs else None, args.out.with_suffix(".pcu.json"))
    pairs = np.array([len(w) - 1 for _, w in g["lad"]])
    out = summarize(g["res"], pairs, refs, args.nboot)
    args.out.write_text(json.dumps(out))
    for name, r in out.items():
        if name == "per_ladder_pairs":
            continue
        extra = "".join(f"; vs {ref} {100 * d['mean']:+.1f} ({100 * d['ci95'][0]:+.1f} to {100 * d['ci95'][1]:+.1f})"
                        for ref, d in r["diff_vs"].items())
        print(f"{name}: correct {r['correct']}/{r['all_pairs']} = {100 * r['correct_of_all']:.1f}% "
              f"({100 * r['ci95'][0]:.1f} to {100 * r['ci95'][1]:.1f}){extra}; slips per wrap {100 * r['slips_per_wrap']:.1f}%")


if __name__ == "__main__":
    main()
