"""E-03: three spiral fits on one PHerc0826 band and PCU's slips-per-wrap eval, on a GPU pod.

    python e03_fit_pod.py --ours relative_windings.json --work /workspace/e03 [--runs A0,PCU,OURS] [--gpus 0]

Mirrors PCU's Kaggle runner (kaggle/pcu-0826b/pcu_0826b.py in
github.com/Jashann/vesuvius-scrolling) so the comparison differs only in the
constraint file:

  A0    tracks only, no relative-winding constraints
  PCU   tracks + PCU's published constraints (pcu_gold_0826_fit_v2.json, 25,771 ladders)
  OURS  tracks + the file passed as --ours

Same villa commit as PCU (6bbe6e2), same band (z 8500 to 9500), same steps
(12,000), same seed (config default 1), same strip-loss weights as PCU's best
run B1w20 (radius 20, dt 10, 840 draws per step). Output: grids_<run>.npz per
run (one (3, rows, cols) array of x, y, z per winding) and the eval line
"<run>: slips S/T = f per wrap" from PCU's exp/e43_eval_gold.py on the eight
held-out slices (gold0826_heldout8.json).

Needs: an NVIDIA GPU with 16 GB (PCU used a Kaggle T4), about 25 GB of disk
for the Lasagna level-2 stores and tracks, network access to the open-data
bucket, dl.ash2txt.org and GitHub. Python 3.14 is installed by uv.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VILLA_SHA = "6bbe6e2"
PCU_SHA = "4f4997a"
Z0, Z1 = 8500, 9500
STEPS = 12000
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
LAS = "PHerc0826/representations/predictions/lasagna/20250821151701-lasagna-20260419180421"
TRACKS = "https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0826/20250821151701/tracks/PHerc0826_20250821151701_surface_m7_L0_th0.2.dbm"
UMBILICUS = f"{S3}/PHerc0826/representations/umbilicus/20250821151701-umbilicus-20260808113303.json"
EMPTY = {"vc_pointcollections_json_version": "1", "collections": {}}
T0 = time.time()


def sh(cmd: str, check: bool = True, timeout: int = 36000) -> int:
    print(f"\n$ {cmd}  [{time.time() - T0:.0f}s]", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r.returncode


def fetch(url: str, path: Path) -> bool:
    """Download once, atomically; returns False on 404."""
    import requests

    if path.exists() and path.stat().st_size > 0:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(f"{path}.part", "wb") as f:
                    for block in r.iter_content(1 << 22):
                        f.write(block)
            os.replace(f"{path}.part", path)
            return True
        except Exception as e:  # network hiccup: retry, then give up loudly
            print("retry", attempt, url, e, flush=True)
            time.sleep(3)
    raise SystemExit(f"could not fetch {url}")


def install(work: Path) -> tuple[Path, Path]:
    """Clone villa and PCU at their pinned commits; uv sync the fitter. Returns (villa, pcu)."""
    villa, pcu = work / "villa", work / "pcu"
    if shutil.which("uv") is None:
        sh("curl -LsSf https://astral.sh/uv/install.sh | sh")
        os.environ["PATH"] = str(Path.home() / ".local/bin") + ":" + os.environ["PATH"]
    if not villa.exists():
        sh(f"git clone https://github.com/ScrollPrize/villa.git {villa} && git -C {villa} checkout -q {VILLA_SHA}")
    if not pcu.exists():
        sh(f"git clone https://github.com/Jashann/vesuvius-scrolling.git {pcu} && git -C {pcu} checkout -q {PCU_SHA}")
    sh(f"cd {villa}/spiral-fitting && uv sync -q")
    sh("pip install -q awscli requests tifffile scipy numpy")
    return villa, pcu


def assemble_dataset(work: Path, villa: Path) -> Path:
    """Lasagna level-2 stores packed into resident pools, tracks, umbilicus."""
    import requests

    ds = work / "ds"
    las = ds / "lasagna_inputs"
    las.mkdir(parents=True, exist_ok=True)
    for ch in ("nx", "ny", "grad_mag"):
        dst = las / f"PHerc0826_{ch}.ome.zarr"
        sh(f"aws s3 sync --no-sign-request --only-show-errors s3://vesuvius-challenge-open-data/{LAS}/PHerc0826_{ch}.ome.zarr/2 {dst}/2")
        for f in (".zattrs", ".zgroup"):
            r = requests.get(f"{S3}/{LAS}/PHerc0826_{ch}.ome.zarr/{f}", timeout=120)
            if r.ok:
                (dst / f).write_bytes(r.content)
    py = villa / "spiral-fitting/.venv/bin/python"
    if not glob.glob(str(las / "*.respool_g2*")):
        sh(f"cd {villa}/spiral-fitting && {py} pack_resident_pools.py {las} --what normals,grad_mag --normal-group 2 --io-threads 16")
    tracks = ds / "tracks"
    index = requests.get(TRACKS + ".vctracks/", timeout=120).text
    vfiles = [n for n in re.findall(r'href="([^"/]+)"', index) if not n.startswith("..")]
    jobs = [(TRACKS, tracks / "PHerc0826.dbm"), (TRACKS + ".crossings.npz", tracks / "PHerc0826.dbm.crossings.npz")]
    jobs += [(f"{TRACKS}.vctracks/{f}", tracks / "PHerc0826.dbm.vctracks" / f) for f in vfiles]
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(lambda a: fetch(*a), jobs))
    print("tracks fetched", sum(results), "of", len(jobs), flush=True)
    (ds / "umbilicus.json").write_text(json.dumps(requests.get(UMBILICUS, timeout=120).json()))
    return ds


def in_band(collections: dict) -> dict:
    return {k: c for k, c in collections.items() if all(Z0 <= p["p"][2] < Z1 for p in c["points"].values())}


def variant(work: Path, ds: Path, name: str, relative: dict) -> Path:
    """A dataset root per run: shared inputs symlinked, its own constraint file."""
    root = work / f"ds_{name}"
    root.mkdir(exist_ok=True)
    for entry in os.listdir(ds):
        if not (root / entry).exists():
            os.symlink(ds / entry, root / entry)
    for f in ("abs_winding.json", "same_windings.json"):
        (root / f).write_text(json.dumps(EMPTY))
    (root / "relative_windings.json").write_text(json.dumps(relative))
    (root / "spiral-scroll.json").write_text(json.dumps({
        "schema_version": 1, "name": f"PHerc0826_{name}", "voxel_size_um": 9.362, "spiral_outward_sense": "CW",
        "normal_zarr_group": "2", "lasagna_scale": 4,
        "paths": {"tracks_dbm": "tracks/PHerc0826.dbm",
                  "normal_x": "lasagna_inputs/PHerc0826_nx.ome.zarr",
                  "normal_y": "lasagna_inputs/PHerc0826_ny.ome.zarr",
                  "gradient_magnitude": "lasagna_inputs/PHerc0826_grad_mag.ome.zarr"}}))
    return root


def fit_config(use_relative: bool) -> dict:
    cfg = {"z_begin": Z0, "z_end": Z1, "optimizer_num_training_steps": STEPS,
           "input_use_tracks": True, "input_use_fibers": False, "input_use_fiber_directions": False,
           "input_use_verified_patches": False, "input_use_winding_inference": False,
           "input_use_pcl_absolute": False, "input_use_pcl_same_winding": False,
           "input_use_pcl_drawn_control_points": False, "input_use_outer_shell": False,
           "input_use_pcl_relative": use_relative}
    if use_relative:
        cfg.update({"sample_count_unattached_pcls_per_step": 840,
                    "loss_weight_unattached_pcl_radius": 20.0, "loss_weight_unattached_pcl_dt": 10.0})
    return cfg


def run_fit(villa: Path, name: str, root: Path, cfg: dict, gpu: str, work: Path) -> subprocess.Popen:
    out = work / f"out_{name}"
    env = (f"CUDA_VISIBLE_DEVICES={gpu} FIT_SPIRAL_OUT_DIR={out} FIT_SPIRAL_CACHE_DIR={work}/cache_{name} "
           f"WANDB_MODE=disabled FIT_SPIRAL_CONFIG_OVERRIDES='{json.dumps(cfg)}'")
    log = open(work / f"fit_{name}.log", "w")
    print(f"\n$ [{name} on GPU {gpu}] {env} fit_spiral.py --dataset {root}", flush=True)
    return subprocess.Popen(f"cd {villa}/spiral-fitting && {env} .venv/bin/python fit_spiral.py --dataset {root}",
                            shell=True, stdout=log, stderr=subprocess.STDOUT)


def stack_grids(work: Path, name: str) -> Path:
    import numpy as np
    import tifffile

    dirs = [d for d in glob.glob(f"{work}/out_{name}/**/w*", recursive=True)
            if re.fullmatch(r"w\d{3}", os.path.basename(d)) and os.path.exists(f"{d}/x.tif")]
    grids = {os.path.basename(d): np.stack([tifffile.imread(f"{d}/{a}.tif") for a in "xyz"]).astype(np.float32) for d in sorted(dirs)}
    path = work / "runs" / "kaggle" / name / f"grids_{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **grids)
    for mf in glob.glob(f"{work}/out_{name}/**/*satisf*.json", recursive=True)[:2]:
        shutil.copy(mf, path.parent / f"{name}_{os.path.basename(mf)}")
    print(name, "windings exported", len(grids), flush=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True, help="our relative_windings.json (level-0, integer keys)")
    ap.add_argument("--work", type=Path, default=Path("/workspace/e03"))
    ap.add_argument("--runs", default="A0,PCU,OURS")
    ap.add_argument("--gpus", default="0", help="comma list; runs are spread over these and run in parallel per GPU")
    ap.add_argument("--skip-fit", action="store_true", help="only stack and evaluate existing outputs")
    args = ap.parse_args()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    villa, pcu = install(work)
    ds = assemble_dataset(work, villa)

    ours = json.loads(args.ours.read_text())
    ours["collections"] = in_band(ours["collections"])
    pcu_file = pcu / "data/gold/pcu_gold_0826_fit_v2.json"
    pcu_rel = json.loads(pcu_file.read_text())
    pcu_rel["collections"] = in_band(pcu_rel["collections"])
    print("constraint ladders in band: ours", len(ours["collections"]), "PCU", len(pcu_rel["collections"]), flush=True)

    runs = {"A0": (variant(work, ds, "A0", EMPTY), fit_config(False)),
            "PCU": (variant(work, ds, "PCU", pcu_rel), fit_config(True)),
            "OURS": (variant(work, ds, "OURS", ours), fit_config(True))}
    runs = {k: v for k, v in runs.items() if k in args.runs.split(",")}
    gpus = args.gpus.split(",")
    if not args.skip_fit:
        # one fit at a time per GPU; the rest queue behind it
        queue = list(runs.items())
        active: dict[str, tuple[str, subprocess.Popen]] = {}
        while queue or active:
            for g in gpus:
                if g not in active and queue:
                    name, (root, cfg) = queue.pop(0)
                    active[g] = (name, run_fit(villa, name, root, cfg, g, work))
            for g, (name, p) in list(active.items()):
                if p.poll() is not None:
                    print(f"fit {name} rc={p.returncode} [{time.time() - T0:.0f}s]", flush=True)
                    if p.returncode != 0:
                        sh(f"tail -40 {work}/fit_{name}.log", check=False)
                    del active[g]
            time.sleep(30)
    for name in runs:
        stack_grids(work, name)
    gold = pcu / "data/gold/gold0826_heldout8.json"
    for name in runs:
        sh(f"cd {pcu} && LADDERS={gold} Z0={Z0} Z1={Z1} python exp/e43_eval_gold.py {name} 2>&1 | tee {work}/eval_{name}.txt",
           check=False)
    print("total", f"{time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
