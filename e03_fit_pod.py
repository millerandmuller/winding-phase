"""E-03: three spiral fits on one PHerc0826 band and PCU's slips-per-wrap eval, on a GPU pod.

    python e03_fit_pod.py --ours relative_windings.json --work /workspace/e03 [--runs A0,PCU,OURS] [--gpus 0]

Named runs (see RUNS): A0, PCU, OURS are the three-way comparison at PCU's
B1w20 settings; the *_s2 runs repeat with optimizer seed 2; OURS_d8400 draws
ten times more ladders per step (our file has ten times PCU's ladders);
OURS_w40 doubles the strip radius weight; OURS_d8400_w40 does both.

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

--scroll PHercParis4 (human-label check): the organizers' published spiral
dataset as PCU used it, PCU's Paris 4 config (weights 10 / 10, outer shell at
the fitter default), band z 8400 to 9400; A0 is PCU's A2, PCU is their B3w10
(pass --pcu-file pcu_gold_Paris4_z8400_9400_fit_input.json), and every run is
scored on the human ladders in the band with PCU's e31b_bootstrap.py.

Needs: an NVIDIA GPU (tested on 24 and 32 GB; PCU used a 16 GB Kaggle T4 for their smaller file), about 25 GB of disk
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
STEPS = 12000
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
EMPTY = {"vc_pointcollections_json_version": "1", "collections": {}}

# Per-scroll inputs (open-data bucket and dl.ash2txt.org; inventory 2026-10-23).
# z0, z1: the band PCU's constraints cover; the catalog flags (z top to bottom,
# right-handed) are the same for both scrolls, so the spiral sense is CW for both.
SCROLLS = {
    "PHerc0826": {"volume": "20250821151701", "lasagna": "20250821151701-lasagna-20260419180421",
                  "tracks": "PHerc0826_20250821151701_surface_m7_L0_th0.2.dbm",
                  "umbilicus": "20250821151701-umbilicus-20260808113303.json", "z0": 8500, "z1": 9500,
                  "pcu_file": "data/gold/pcu_gold_0826_fit_v2.json", "gold_file": "data/gold/gold0826_heldout8.json"},
    "PHerc0211": {"volume": "20250821151803", "lasagna": "20250821151803-lasagna-20260419180421",
                  "tracks": "PHerc0211_20250821151803_surface_m7_L0_th0.2.dbm",
                  "umbilicus": "20250821151803-umbilicus-20260808112626.json", "z0": 9500, "z1": 10500,
                  "pcu_file": None, "gold_file": None},  # pass --pcu-file and --gold (from e03_split_gold.py)
    # Human-label check only (not eligible). The organizers' spiral dataset is
    # used as published, exactly as PCU's modal/fit_p4.py does; scoring is on
    # the human ladders in the band (PCU's exp/e31b_bootstrap.py "human").
    "PHercParis4": {"dataset": "spiral", "z0": 8400, "z1": 9400, "pcu_file": None, "gold_file": None},
}
P4_DL = "https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4"
P4_TRACKS = "2um_ds2_ps256_surf_v2.dbm"
# Files the per-run dataset writes itself (never symlinked: the Paris 4
# dataset ships human versions of the first three).
PER_RUN_FILES = ("relative_windings.json", "abs_winding.json", "same_windings.json", "spiral-scroll.json")
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
    # Ubuntu 24.04 images mark the system Python as externally managed (PEP 668);
    # the pod is a disposable container, so installing into it is fine.
    sh("pip install -q --break-system-packages awscli requests tifffile scipy numpy 'zarr<3' fsspec aiohttp")  # PCU's eval imports zarr 2's FSStore
    return villa, pcu


def assemble_dataset(work: Path, villa: Path, scroll: str, sc: dict) -> Path:
    """Lasagna level-2 stores packed into resident pools, tracks, umbilicus."""
    import requests

    las_root = f"{scroll}/representations/predictions/lasagna/{sc['lasagna']}"
    tracks_url = f"https://dl.ash2txt.org/datasets/spiral_datasets/{scroll}/{sc['volume']}/tracks/{sc['tracks']}"
    umbilicus_url = f"{S3}/{scroll}/representations/umbilicus/{sc['umbilicus']}"
    ds = work / "ds"
    las = ds / "lasagna_inputs"
    las.mkdir(parents=True, exist_ok=True)
    packed = bool(glob.glob(str(las / "*.respool_g2*")))
    if not packed:
        for ch in ("nx", "ny", "grad_mag"):
            dst = las / f"{scroll}_{ch}.ome.zarr"
            sh(f"aws s3 sync --no-sign-request --only-show-errors s3://vesuvius-challenge-open-data/{las_root}/{scroll}_{ch}.ome.zarr/2 {dst}/2")
            for f in (".zattrs", ".zgroup"):
                r = requests.get(f"{S3}/{las_root}/{scroll}_{ch}.ome.zarr/{f}", timeout=120)
                if r.ok:
                    (dst / f).write_bytes(r.content)
        py = villa / "spiral-fitting/.venv/bin/python"
        sh(f"cd {villa}/spiral-fitting && {py} pack_resident_pools.py {las} --what normals,grad_mag --normal-group 2 --io-threads 16")
        # The fitter reads the normals and grad_mag only through the packed
        # sidecars; the raw chunks (about 8 GB) would otherwise fill a 50 GB pod.
        for ch in ("nx", "ny", "grad_mag"):
            shutil.rmtree(las / f"{scroll}_{ch}.ome.zarr" / "2", ignore_errors=True)
        shutil.rmtree(Path.home() / ".cache/uv", ignore_errors=True)
    tracks = ds / "tracks"
    index = requests.get(tracks_url + ".vctracks/", timeout=120).text
    vfiles = [n for n in re.findall(r'href="([^"/]+)"', index) if not n.startswith("..")]
    jobs = [(tracks_url, tracks / f"{scroll}.dbm"), (tracks_url + ".crossings.npz", tracks / f"{scroll}.dbm.crossings.npz")]
    jobs += [(f"{tracks_url}.vctracks/{f}", tracks / f"{scroll}.dbm.vctracks" / f) for f in vfiles]
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(lambda a: fetch(*a), jobs))
    print("tracks fetched", sum(results), "of", len(jobs), flush=True)
    (ds / "umbilicus.json").write_text(json.dumps(requests.get(umbilicus_url, timeout=120).json()))
    return ds


def assemble_spiral_dataset(work: Path) -> Path:
    """The published Paris 4 spiral dataset, the same file list as PCU's modal/fit_p4.py prep()."""
    ds = work / "ds"
    jobs = [(f"{P4_DL}/{f}", ds / f) for f in ("umbilicus.json", "abs_winding.json", "relative_windings.json", "same_windings.json")]
    jobs += [(f"{P4_DL}/outer_shell/{f}", ds / "outer_shell" / f) for f in ("meta.json", "x.tif", "y.tif", "z.tif")]
    for store, files in (("las_008_grad_mag.ome.zarr.respool_g4", ("meta.json", "table.npy", "brick_coords.npy", "channel_0.u8")),
                         ("las_008_nx.ome.zarr.respool_g4_pair", ("meta.json", "table.npy", "brick_coords.npy", "channel_0.u8", "channel_1.u8"))):
        jobs += [(f"{P4_DL}/lasagna_inputs/{store}/{f}", ds / "lasagna_inputs" / store / f) for f in files]
    jobs += [(f"{P4_DL}/lasagna_inputs/{f}", ds / "lasagna_inputs" / f) for f in ("verification.json", "mask_report.json", "batch_verification.json")]
    tb = f"{P4_DL}/tracks/{P4_TRACKS}"
    jobs += [(f"{tb}.db", ds / "tracks" / f"{P4_TRACKS}.db"), (f"{tb}.crossings.npz", ds / "tracks" / f"{P4_TRACKS}.crossings.npz")]
    jobs += [(f"{tb}.vctracks/{f}", ds / "tracks" / f"{P4_TRACKS}.vctracks" / f) for f in
             ("arclengths.f64", "coordinates.i32", "family_codes.i8", "header.bin", "metadata.json", "offsets.i64",
              "source_ids.u64", "tortuosities.f64", "z_bounds.i32")]
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(lambda a: fetch(*a), jobs))
    print("spiral dataset files fetched", sum(results), "of", len(jobs), flush=True)  # verification jsons may 404
    (ds / "verified_patches").mkdir(exist_ok=True)
    return ds


def in_band(collections: dict, z0: int, z1: int) -> dict:
    return {k: c for k, c in collections.items() if all(z0 <= p["p"][2] < z1 for p in c["points"].values())}


def variant(work: Path, ds: Path, name: str, relative: dict, scroll: str) -> Path:
    """A dataset root per run: shared inputs symlinked, its own constraint file."""
    root = work / f"ds_{name}"
    root.mkdir(exist_ok=True)
    for entry in os.listdir(ds):
        if entry not in PER_RUN_FILES and not (root / entry).exists():
            os.symlink(ds / entry, root / entry)
    for f in ("abs_winding.json", "same_windings.json"):
        (root / f).write_text(json.dumps(EMPTY))
    (root / "relative_windings.json").write_text(json.dumps(relative))
    if SCROLLS[scroll].get("dataset") == "spiral":
        # PCU's Paris 4 scroll file: conventional paths, 9.6 um voxels.
        (root / "spiral-scroll.json").write_text(json.dumps({
            "schema_version": 1, "name": f"P4{name}", "voxel_size_um": 9.6, "spiral_outward_sense": "CW"}))
        return root
    (root / "spiral-scroll.json").write_text(json.dumps({
        "schema_version": 1, "name": f"{scroll}_{name}", "voxel_size_um": 9.362, "spiral_outward_sense": "CW",
        "normal_zarr_group": "2", "lasagna_scale": 4,
        "paths": {"tracks_dbm": f"tracks/{scroll}.dbm",
                  "normal_x": f"lasagna_inputs/{scroll}_nx.ome.zarr",
                  "normal_y": f"lasagna_inputs/{scroll}_ny.ome.zarr",
                  "gradient_magnitude": f"lasagna_inputs/{scroll}_grad_mag.ome.zarr"}}))
    return root


# name -> (constraint source, config overrides on top of fit_config)
RUNS = {
    "A0": (None, {}),
    "PCU": ("pcu", {}),
    "OURS": ("ours", {}),
    "PCU_s2": ("pcu", {"optimizer_random_seed": 2}),
    "OURS_s2": ("ours", {"optimizer_random_seed": 2}),
    "A0_s2": (None, {"optimizer_random_seed": 2}),
    "OURS_d8400": ("ours", {"sample_count_unattached_pcls_per_step": 8400}),
    "OURS_w40": ("ours", {"loss_weight_unattached_pcl_radius": 40.0}),
    "OURS_d8400_w40": ("ours", {"sample_count_unattached_pcls_per_step": 8400, "loss_weight_unattached_pcl_radius": 40.0}),
    # the same two settings on PCU's file, so a gain is not an artefact of the settings
    "PCU_w40": ("pcu", {"loss_weight_unattached_pcl_radius": 40.0}),
    "PCU_d8400_w40": ("pcu", {"sample_count_unattached_pcls_per_step": 8400, "loss_weight_unattached_pcl_radius": 40.0}),
}


def fit_config_p4(use_relative: bool, z0: int, z1: int) -> dict:
    """PCU's Paris 4 config (modal/fit_p4.py, run B3w10): outer shell left at the fitter default (on)."""
    cfg = {"z_begin": z0, "z_end": z1, "optimizer_num_training_steps": STEPS,
           "input_use_fibers": False, "input_use_pcl_drawn_control_points": False, "input_use_fiber_directions": False,
           "input_use_tracks": True, "input_use_verified_patches": False,
           "input_use_pcl_relative": use_relative, "input_use_pcl_absolute": False, "input_use_pcl_same_winding": False,
           "input_use_winding_inference": False}
    if use_relative:
        cfg.update({"sample_count_unattached_pcls_per_step": 840,
                    "loss_weight_unattached_pcl_radius": 10.0, "loss_weight_unattached_pcl_dt": 10.0})
    return cfg


def fit_config(use_relative: bool, z0: int, z1: int) -> dict:
    cfg = {"z_begin": z0, "z_end": z1, "optimizer_num_training_steps": STEPS,
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
    for ck in glob.glob(f"{work}/out_{name}/**/checkpoint_*.ckpt", recursive=True):
        os.remove(ck)  # about 1 GB each; the grids are the product
    print(name, "windings exported", len(grids), flush=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, default=None, help="our relative_windings.json (level-0, integer keys); not needed for A0 only")
    ap.add_argument("--scroll", default="PHerc0826", choices=sorted(SCROLLS))
    ap.add_argument("--pcu-file", type=Path, default=None, help="PCU's constraint file; default from the PCU repo for PHerc0826")
    ap.add_argument("--gold", type=Path, default=None, help="held-out gold ladders for the eval; default from the PCU repo for PHerc0826")
    ap.add_argument("--work", type=Path, default=Path("/workspace/e03"))
    ap.add_argument("--runs", default="A0,PCU,OURS")
    ap.add_argument("--gpus", default="0", help="comma list; runs are spread over these and run in parallel per GPU")
    ap.add_argument("--skip-fit", action="store_true", help="only stack and evaluate existing outputs")
    args = ap.parse_args()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    sc = SCROLLS[args.scroll]
    z0, z1 = sc["z0"], sc["z1"]
    villa, pcu = install(work)
    spiral = sc.get("dataset") == "spiral"
    ds = assemble_spiral_dataset(work) if spiral else assemble_dataset(work, villa, args.scroll, sc)
    base_config = fit_config_p4 if spiral else fit_config

    wanted = {RUNS[n][0] for n in args.runs.split(",")}
    sources = {None: EMPTY}
    if "ours" in wanted:
        ours = json.loads(args.ours.read_text())
        ours["collections"] = in_band(ours["collections"], z0, z1)
        sources["ours"] = ours
    if "pcu" in wanted:
        pcu_file = args.pcu_file or (pcu / sc["pcu_file"] if sc["pcu_file"] else None)
        if pcu_file is None:
            raise SystemExit(f"--pcu-file is required for {args.scroll}")
        pcu_rel = json.loads(Path(pcu_file).read_text())
        pcu_rel["collections"] = in_band(pcu_rel["collections"], z0, z1)
        sources["pcu"] = pcu_rel
    print("constraint ladders in band:", {k: len(v["collections"]) for k, v in sources.items() if k}, flush=True)

    runs = {}
    for name in args.runs.split(","):
        source, extra = RUNS[name]
        runs[name] = (variant(work, ds, name, sources[source], args.scroll), dict(base_config(source is not None, z0, z1), **extra))
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
    if spiral:
        # human ladders in the band, fetched by PCU's script from the dataset; reference A0 when present
        fits = " ".join(f"{n}={work}/runs/kaggle/{n}/grids_{n}.npz" for n in runs)
        ref = "--ref A0" if "A0" in runs else ""
        sh(f"cd {pcu} && Z0={z0} Z1={z1} python exp/e31b_bootstrap.py human {fits} {ref} --out {work}/eval_human.json "
           f"2>&1 | tee {work}/eval_human.txt", check=False)
        print("total", f"{time.time() - T0:.0f}s", flush=True)
        return
    gold = args.gold or (pcu / sc["gold_file"] if sc["gold_file"] else None)
    if gold is None:
        raise SystemExit(f"--gold is required for {args.scroll}")
    for name in runs:
        # the eval globs runs/kaggle/<run>/grids_*.npz relative to the working directory
        sh(f"cd {work} && LADDERS={gold} Z0={z0} Z1={z1} python {pcu}/exp/e43_eval_gold.py {name} 2>&1 | tee {work}/eval_{name}.txt",
           check=False)
    print("total", f"{time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
