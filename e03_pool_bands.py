"""Pool correct-of-all pairs over several bands, with a paired, band-stratified bootstrap.

    python e03_pool_bands.py band1.json band2.json --runs OURS_d8400_w40,OURS_d8400_w40_s2,PCU --ref A0 --out pooled.json

Inputs are outputs of e03_correct_of_all.py (one per band) that share run names. Each
resample draws ladders with replacement within each band separately, so every band keeps
its own number of ladders; the pooled score is correct pairs over all pairs in all bands.
Generator seed 1 and 2,000 resamples, as in PCU's e31b_bootstrap.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_band(path: Path, names: list[str]) -> dict:
    d = json.loads(path.read_text())
    missing = [n for n in names if n not in d]
    if missing:
        raise SystemExit(f"{path}: runs not found: {missing}")
    band = {"N": np.array(d["per_ladder_pairs"])}
    for n in names:
        band[n] = np.array(d[n]["per_ladder_assigned"]) - np.array(d[n]["per_ladder_slips"])
    return band


def pooled_rate(bands: list[dict], name: str, draws: list[np.ndarray]) -> float:
    correct = sum(b[name][i].sum() for b, i in zip(bands, draws))
    total = sum(b["N"][i].sum() for b, i in zip(bands, draws))
    return float(correct / max(total, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bands", nargs="+", type=Path)
    ap.add_argument("--runs", required=True, help="comma list of runs to report")
    ap.add_argument("--ref", required=True, help="reference run for the paired differences")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runs = args.runs.split(",")
    names = sorted(set(runs) | {args.ref})
    bands = [load_band(p, names) for p in args.bands]
    rng = np.random.default_rng(1)
    boot = [[rng.integers(0, len(b["N"]), len(b["N"])) for b in bands] for _ in range(args.nboot)]
    every = [np.arange(len(b["N"])) for b in bands]
    ref_boot = np.array([pooled_rate(bands, args.ref, d) for d in boot])
    out = {"bands": [str(p) for p in args.bands], "ladders": [len(b["N"]) for b in bands],
           "pairs": int(sum(b["N"].sum() for b in bands)), "ref": args.ref,
           args.ref: {"correct_of_all": pooled_rate(bands, args.ref, every)}}
    for name in runs:
        vals = np.array([pooled_rate(bands, name, d) for d in boot])
        diff = vals - ref_boot
        out[name] = {"correct_of_all": pooled_rate(bands, name, every),
                     "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                     "diff_vs_ref": pooled_rate(bands, name, every) - out[args.ref]["correct_of_all"],
                     "diff_ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    args.out.write_text(json.dumps(out, indent=1))
    print(f"pooled: {out['ladders']} ladders, {out['pairs']} pairs; reference {args.ref} "
          f"{100 * out[args.ref]['correct_of_all']:.1f}%")
    for name in runs:
        r = out[name]
        print(f"{name}: {100 * r['correct_of_all']:.1f}% ({100 * r['ci95'][0]:.1f} to {100 * r['ci95'][1]:.1f}); "
              f"vs {args.ref} {100 * r['diff_vs_ref']:+.1f} ({100 * r['diff_ci95'][0]:+.1f} to {100 * r['diff_ci95'][1]:+.1f})")


if __name__ == "__main__":
    main()
