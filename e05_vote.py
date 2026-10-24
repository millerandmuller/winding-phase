"""Track-agreement vote: keep the ladder hops a tracks-only fit agrees with.

    python e05_vote.py relative_windings.json --grid grids_A0.npz --out relative_windings_vote.json

Input is a signed, merged band file (e03_sign.py then e03_merge_band.py): each
ladder runs in the direction of increasing winding, one winding per hop. For
every hop, the winding index of the nearest surface of the tracks-only grid is
read at each end, within --radius level-0 voxels at that slice (the grid is
read as e03_sign.py reads it). A hop is dropped when both ends are read and the
index difference is not +1; it is kept when either end has no surface in reach.
Ladders are split at dropped hops and pieces with a single point are dropped.
The output keeps the fitter's format, including the version key.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from e03_sign import grid_curves_at_z


def read_windings(yx: np.ndarray, tree: cKDTree, winds: np.ndarray, radius: float) -> np.ndarray:
    """Winding index of the nearest grid surface point for each (y, x), NaN where none is in reach."""
    d, idx = tree.query(yx, distance_upper_bound=radius)
    return np.where(np.isfinite(d), winds[np.minimum(idx, len(winds) - 1)], np.nan)


def hop_verdicts(w: np.ndarray) -> np.ndarray:
    """Per hop: 1 agrees (+1 step), 0 unread (an end has no surface), -1 disagrees."""
    dw = np.diff(w)
    read = np.isfinite(dw)
    return np.where(~read, 0, np.where(dw == 1, 1, -1))


def split_at_drops(points: list, verdicts: np.ndarray) -> list[list]:
    """Pieces of a ladder between dropped hops, each with at least two points."""
    pieces, cur = [], [points[0]]
    for p, v in zip(points[1:], verdicts):
        if v == -1:
            pieces.append(cur)
            cur = [p]
        else:
            cur.append(p)
    pieces.append(cur)
    return [piece for piece in pieces if len(piece) >= 2]


def group_by_z(cols: dict) -> dict[float, list]:
    by_z = defaultdict(list)
    for c in cols.values():
        points = [p for _, p in sorted(c["points"].items(), key=lambda kv: int(kv[0]))]
        by_z[float(points[0]["p"][2])].append((c, points))
    return by_z


def vote(cols: dict, grid, radius: float) -> tuple[dict, dict]:
    """Voted collections and counts."""
    out, counts = {}, defaultdict(int)
    for z0, ladders in sorted(group_by_z(cols).items()):
        pts, winds = grid_curves_at_z(grid, z0)
        tree = cKDTree(pts)
        n = 0
        for c, points in ladders:
            yx = np.array([[p["p"][1], p["p"][0]] for p in points])
            v = hop_verdicts(read_windings(yx, tree, winds, radius))
            counts["ladders_in"] += 1
            counts["hops_in"] += len(v)
            counts["hops_agree"] += int((v == 1).sum())
            counts["hops_unread"] += int((v == 0).sum())
            counts["hops_dropped"] += int((v == -1).sum())
            for piece in split_at_drops(points, v):
                n += 1
                meta = dict(c["metadata"]); meta["vote"] = f"track agreement, radius {radius:g}"
                out[str(int(z0) * 100000 + n)] = {
                    "name": f"{c['name']}_v{n}", "metadata": meta,
                    "points": {str(j + 1): {"p": p["p"], "wind_a": float(j + 1)} for j, p in enumerate(piece)}}
                counts["ladders_out"] += 1
                counts["hops_out"] += len(piece) - 1
    return out, dict(counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("band", type=Path, help="signed, merged band file (relative_windings.json format)")
    ap.add_argument("--grid", type=Path, required=True, help="tracks-only fit grid, grids_<run>.npz")
    ap.add_argument("--radius", type=float, default=120.0, help="level-0 voxels")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cols = json.loads(args.band.read_text())["collections"]
    out, counts = vote(cols, np.load(args.grid), args.radius)
    args.out.write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": out}))
    summary = {"band": args.band.name, "grid": args.grid.name, "radius": args.radius, **counts}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
