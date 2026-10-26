"""Which scorer is right on the label points only one of them assigns (E-06 metric).

    python e06_settle.py --bands B3 B4 B5 B6 B7 --points out/e06/points_{band}.json \
        --sfe out/e06/sfe_{band}.json --out figures/e06/settle.json

Inputs per band: the per-point records of e06_exact_distance.py (e31b as written and the exact measure) and
spiralfiteval's winding per point (``sfe_w`` = floor of its spiral time, ``null`` where it finds no sheet
within 10 voxels along the normal), in the same point order.

Order of work: first the agreement check on points both scorers assign (d_exact < 10 and the exact winding
equal to both scorers' winding in at least 95% of instances). If it fails, nothing else is computed. Then:
S = share of disputed instances (spiralfiteval assigns, e31b as written does not) with d_exact < 10 on
spiralfiteval's winding; R = the same for the reverse set, on e31b's winding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CUTOFF = 10.0
AGREEMENT_MIN = 0.95
SAMPLE = 20


def finite(x) -> float:
    return np.inf if x is None else float(x)


def load_band(points_path: Path, sfe_path: Path) -> list[dict]:
    """One record per (fit, point), merging the two files by position."""
    P, S = json.loads(points_path.read_text()), json.loads(sfe_path.read_text())
    recs = []
    for fit, rows in P["fits"].items():
        if len(rows) != len(S[fit]):
            raise ValueError(f"{points_path.name} {fit}: {len(rows)} points, spiralfiteval {len(S[fit])}")
        for r, s in zip(rows, S[fit]):
            recs.append({**r, **s, "fit": fit})
    return recs


def classify(r: dict) -> str:
    e_in, s_in = finite(r["e31b_d"]) < CUTOFF, r["sfe_w"] is not None
    if e_in and s_in:
        return "both"
    if s_in:
        return "only_sfe"
    if e_in:
        return "only_e31b"
    return "neither"


def on_sheet(r: dict, w) -> bool:
    return finite(r["exact_d"]) < CUTOFF and r["exact_w"] == w


def share(n: int, d: int) -> dict:
    return {"n": n, "of": d, "share": (n / d) if d else None}


def agreement(recs: list[dict]) -> dict:
    both = [r for r in recs if r["class"] == "both"]
    ok = sum(on_sheet(r, r["sfe_w"]) and r["sfe_w"] == r["e31b_w"] for r in both)
    return share(ok, len(both))


def settle(recs: list[dict]) -> dict:
    only_s = [r for r in recs if r["class"] == "only_sfe"]
    only_e = [r for r in recs if r["class"] == "only_e31b"]
    S = share(sum(on_sheet(r, r["sfe_w"]) for r in only_s), len(only_s))
    R = share(sum(on_sheet(r, r["e31b_w"]) for r in only_e), len(only_e))
    off = [r for r in only_s if finite(r["exact_d"]) >= CUTOFF]
    other_w = [r for r in only_s if finite(r["exact_d"]) < CUTOFF and r["exact_w"] != r["sfe_w"]]
    d = np.array([finite(r["exact_d"]) for r in only_s])
    return {"S": S, "R": R,
            "disputed_off_sheet": len(off), "disputed_on_other_winding": len(other_w),
            "disputed_border_invalid": share(sum(bool(r["exact_border"]) for r in only_s), len(only_s)),
            "disputed_exact_d_quantiles_10_50_90": [float(x) for x in np.percentile(d[np.isfinite(d)], [10, 50, 90])]
            if np.isfinite(d).any() else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="+", required=True)
    ap.add_argument("--points", required=True, help="path pattern with {band}")
    ap.add_argument("--sfe", required=True, help="path pattern with {band}")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = []
    for b in args.bands:
        for r in load_band(Path(args.points.format(band=b)), Path(args.sfe.format(band=b))):
            r["band"] = b
            r["class"] = classify(r)
            recs.append(r)

    agree = agreement(recs)
    result = {"instances": len(recs), "agreement_on_both": agree,
              "classes": {c: sum(r["class"] == c for r in recs) for c in ("both", "only_sfe", "only_e31b", "neither")}}
    print(f"agreement on points both scorers assign: {agree['n']}/{agree['of']} ({agree['share']})")
    if agree["share"] is None or agree["share"] < AGREEMENT_MIN:
        result["verdict"] = "KILL: agreement check failed"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(result["verdict"])
        return 1

    result["pooled"] = settle(recs)
    result["per_band"] = {b: settle([r for r in recs if r["band"] == b]) for b in args.bands}
    fits = sorted({r["fit"] for r in recs})
    result["per_fit"] = {f"{b} {f}": settle([r for r in recs if r["band"] == b and r["fit"] == f])
                         for b in args.bands for f in fits}
    only_s = [r for r in recs if r["class"] == "only_sfe"]
    rng = np.random.default_rng(1)
    pick = rng.choice(len(only_s), size=min(SAMPLE, len(only_s)), replace=False) if only_s else []
    result["plot_sample"] = [{k: only_s[i][k] for k in ("band", "fit", "ladder", "point", "sfe_w", "exact_w", "exact_d", "e31b_d")}
                             for i in sorted(pick)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    p = result["pooled"]
    print(f"S = {p['S']['n']}/{p['S']['of']} ({p['S']['share']}); R = {p['R']['n']}/{p['R']['of']} ({p['R']['share']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
