"""E-03: merge per-slice signed ladder files into one relative_windings.json for the fitter.

    python e03_merge_band.py out/band/e03_signed_z*_three_b0.35_full.json --out out/relative_windings.json

Collection keys are already unique across slices (z0 * 100000 + index, see
e03_ladders.py); this checks that, drops nothing, and prints per-slice and
total counts so the file's contents are on record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    merged, per_slice = {}, {}
    for f in sorted(args.files):
        cols = json.loads(f.read_text())["collections"]
        for key, c in cols.items():
            if key in merged:
                raise SystemExit(f"duplicate collection key {key} in {f}")
            int(key)  # the fitter parses keys as integers
            merged[key] = c
        z = next(iter(cols.values()))["points"]["1"]["p"][2] if cols else None
        per_slice[str(z)] = {"ladders": len(cols), "points": sum(len(c["points"]) for c in cols.values())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": merged}))
    summary = {"slices": len(per_slice), "ladders": len(merged), "points": sum(len(c["points"]) for c in merged.values()),
               "sign": {}, "per_slice": per_slice}
    for c in merged.values():
        s = c["metadata"].get("sign", "none"); summary["sign"][s] = summary["sign"].get(s, 0) + 1
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_slice"}, indent=2))


if __name__ == "__main__":
    main()
