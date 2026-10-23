"""E-03: hold out every n-th slice of a PCU gold band file for the eval.

    python e03_split_gold.py pcu_gold_0211.json --every 8 --offset 3 --out out/0211/

PCU published PHerc0826 as a fit file plus a separate held-out set; for
PHerc0211 there is one band file (63 slices every 16 voxels). This writes
`pcu_<tag>_fit.json` (the slices the fitter sees, integer keys kept) and
`gold<tag>_heldout.json` (the held-out slices, keys prefixed with h, for
PCU's exp/e43_eval_gold.py), and prints the slice lists so the split is on
record. Our own constraints are generated on the fit slices only, so both
constraint sets see the same z.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def slice_of(collection: dict) -> int:
    return round(next(iter(collection["points"].values()))["p"][2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold", type=Path)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--offset", type=int, default=3)
    ap.add_argument("--tag", default=None, help="file tag, default from the gold file name")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = json.loads(args.gold.read_text())
    cols = d["collections"]
    slices = sorted({slice_of(c) for c in cols.values()})
    held = set(slices[args.offset :: args.every])
    fit = {k: c for k, c in cols.items() if slice_of(c) not in held}
    heldout = {f"h{k}": c for k, c in cols.items() if slice_of(c) in held}
    tag = args.tag or args.gold.stem.replace("pcu_gold_", "")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"pcu_{tag}_fit.json").write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": fit}))
    (args.out / f"gold{tag}_heldout.json").write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": heldout}))
    summary = {"slices_total": len(slices), "fit_slices": sorted(set(slices) - held), "heldout_slices": sorted(held),
               "fit_ladders": len(fit), "heldout_ladders": len(heldout),
               "fit_points": sum(len(c["points"]) for c in fit.values()), "heldout_points": sum(len(c["points"]) for c in heldout.values())}
    (args.out / f"split_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
