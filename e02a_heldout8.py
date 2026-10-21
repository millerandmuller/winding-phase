"""E-02a: pool the k-wrap certificate scores over all eight PCU held-out slices.

    python e02a_heldout8.py out/e02a_kwrap_z*_full.json --out out/

Reads the per-slice JSON written by e02a_kwrap.py and reports, per slice and
pooled, the accuracy of each estimate per k and the coverage and precision
of each certificate. Pooling is by counts (accepted and correct pairs summed
over slices), not by averaging per-slice rates, so large slices weigh more.
Also lists the raw one-wrap density median per slice, which says whether the
single calibration factor holds along z.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CERTS = ["unwrap+ridges", "unwrap+density", "all three",
         "ridges+density scored on ridges (no unwrap needed)"]


def counts_from_rate(rate: float | None, n: int) -> int:
    """Recover a count from a rate rounded to 4 decimals (exact for n below 5,000)."""
    return int(round(rate * n)) if rate is not None else 0


def slice_row(d: dict) -> dict:
    row = {"z0": d["z0"], "pairs": d["pairs_total"], "median_density_k1": d["median_raw_density_integral_k1"]}
    for k, pk in d["per_k"].items():
        n = pk["pairs"]
        unw_n = counts_from_rate(pk["unwrap_covered"], n)
        row[f"k{k}"] = {"pairs": n,
                        "unwrap_correct": counts_from_rate(pk["unwrap_acc"], unw_n), "unwrap_covered": unw_n,
                        "ridges_correct": counts_from_rate(pk["ridges_acc"], n),
                        "density_correct": counts_from_rate(pk["density_acc"], n)}
    for c in CERTS:
        cert = d[c]
        row[c] = {"n": cert["n"], "correct": counts_from_rate(cert["precision"], cert["n"])}
    return row


def pool(rows: list[dict]) -> dict:
    total = sum(r["pairs"] for r in rows)
    out = {"slices": [r["z0"] for r in rows], "pairs": total, "per_k": {}, "certificates": {}}
    ks = sorted({k for r in rows for k in r if k.startswith("k")})
    for k in ks:
        n = sum(r[k]["pairs"] for r in rows if k in r)
        unw_n = sum(r[k]["unwrap_covered"] for r in rows if k in r)
        out["per_k"][k] = {"pairs": n,
                           "unwrap_acc": round(sum(r[k]["unwrap_correct"] for r in rows if k in r) / unw_n, 4) if unw_n else None,
                           "unwrap_covered": round(unw_n / n, 4),
                           "ridges_acc": round(sum(r[k]["ridges_correct"] for r in rows if k in r) / n, 4),
                           "density_acc": round(sum(r[k]["density_correct"] for r in rows if k in r) / n, 4)}
    for c in CERTS:
        n = sum(r[c]["n"] for r in rows)
        ok = sum(r[c]["correct"] for r in rows)
        out["certificates"][c] = {"coverage": round(n / total, 4), "precision": round(ok / n, 4) if n else None, "n": n, "correct": ok}
    meds = [r["median_density_k1"] for r in rows if r["median_density_k1"] is not None]
    out["median_density_k1_by_slice"] = {r["z0"]: r["median_density_k1"] for r in rows}
    out["median_density_k1_range"] = [min(meds), max(meds)] if meds else None
    return out


def fmt(ok: int, n: int, total: int) -> str:
    return f"{100 * ok / n:5.1f}% at {100 * n / total:4.1f}%" if n else "none"


def print_table(rows: list[dict], pooled: dict) -> None:
    print(f"{'z0':>6} {'pairs':>6} {'dens.med':>8} | " + " | ".join(f"{c[:18]:>18}" for c in CERTS))
    for r in rows:
        cells = [fmt(r[c]["correct"], r[c]["n"], r["pairs"]) for c in CERTS]
        print(f"{r['z0']:>6} {r['pairs']:>6} {r['median_density_k1']:>8.3f} | " + " | ".join(f"{s:>18}" for s in cells))
    cells = [fmt(pooled["certificates"][c]["correct"], pooled["certificates"][c]["n"], pooled["pairs"]) for c in CERTS]
    print(f"{'pooled':>6} {pooled['pairs']:>6} {'':>8} | " + " | ".join(f"{s:>18}" for s in cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()
    rows = sorted((slice_row(json.loads(p.read_text())) for p in args.jsons), key=lambda r: r["z0"])
    pooled = pool(rows)
    print_table(rows, pooled)
    print(json.dumps(pooled, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "e02a_heldout8_pooled.json").write_text(json.dumps({"per_slice": rows, "pooled": pooled}, indent=2))


if __name__ == "__main__":
    main()
