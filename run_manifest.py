"""Write a run manifest: code versions, packages, input hashes, seed and command line.

    python run_manifest.py --run-id p4_B3_OURS_d8400_w40 --seed 1 --cmd "python3 e03_fit_pod.py ..." \
        --input ours=out/e05/B3/relative_windings_ours_B3.json --output grid=out/e05/B3/grids/grids_OURS_d8400_w40.npz \
        --env out/e05/manifest/pod_env.txt --packages out/e05/manifest/fitter_packages.txt --out runs/p4_B3_OURS_d8400_w40/manifest.json

The product commit is read from git here; anything about the machine that ran the fit
(villa and PCU commits, GPU, package list) comes from the files passed with --env and
--packages, captured on that machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def named_files(items: list[str]) -> dict:
    """NAME=path pairs -> {name: {path, sha256, bytes}}; a missing file is an error."""
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        p = Path(path)
        if not name or not path:
            raise SystemExit(f"expected NAME=path, got {item!r}")
        if not p.is_file():
            raise SystemExit(f"not a file: {p}")
        out[name] = {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
    return out


def git_state(repo: Path) -> dict:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": head, "uncommitted_changes": bool(dirty)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cmd", required=True, help="the command line that produced the outputs")
    ap.add_argument("--config", default="{}", help="JSON of settings that differ from the defaults")
    ap.add_argument("--input", action="append", default=[], help="NAME=path, repeatable")
    ap.add_argument("--output", action="append", default=[], help="NAME=path, repeatable")
    ap.add_argument("--env", type=Path, action="append", default=[], help="text file captured on the run machine")
    ap.add_argument("--packages", type=Path, action="append", default=[], help="frozen package list from the run machine")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = {
        "run_id": args.run_id,
        "product": git_state(Path(__file__).resolve().parent),
        "seed": args.seed,
        "command": args.cmd,
        "config": json.loads(args.config),
        "inputs": named_files(args.input),
        "outputs": named_files(args.output),
        "environment": {p.name: p.read_text().splitlines() for p in args.env},
        "packages": {p.name: p.read_text().splitlines() for p in args.packages},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
