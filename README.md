# winding-phase

**Can Lasagna's cosine channel be treated as a wrapped phase and unwrapped, InSAR style, into winding constraints for the villa spiral fitter?**

Status, 2026-10-24. On one band of PHerc0826 (z 8500 to 9500), relative-winding constraints generated from one Lasagna run take villa's spiral fitter from 49.5% to 32.1% slips per wrap on PCU's held-out ladders (95% interval 30.7 to 33.6), against 36.1% for PCU's published constraint set at its best. Two caveats carry through the whole README. The scoring ladders are PCU's, built from the same Lasagna output; they measure agreement with PCU's reading of Lasagna, not with human-traced windings. And the result is one band of one scroll. A second scroll (PHerc0211) and a check against human labels (Paris 4) are the next experiments.

## What is here

| File | Does |
|---|---|
| `lasagna_store.py` | Reads the published Lasagna pyramids (cos, grad_mag, nx, ny) and the CT for one scroll from the open-data bucket, one slice window at a time; decodes the uint8 encodings |
| `phase.py` | Wrapped phase per slice: local contrast normalisation, Riesz-transform quadrature projected on the predicted normal, `atan2`; 2x2 loop residues |
| `umbilicus.py` | Outward reference direction from the hand-annotated umbilicus, used to orient the normals |
| `e01_residue_map.py` | One slice: three-panel figure (CT, wrapped phase, residues over CT) plus a JSON of the numbers |
| `e01_variants.py` | Same crop under three settings, to see what drives the residue count |
| `e01_residue_analysis.py` | Residue density by local fringe period and by distance from the umbilicus; dipole share |
| `e01_papyrus_split.py` | Residues on vs off papyrus, with the m7 surface prediction as the mask |
| `e01_zoom.py` | Zoomed windows at native pixels, residues over the CT |
| `e02a_baseline.py` | 2D unwrap of one held-out slice, scored on PCU's gold ladders |
| `e02a_gates.py` | Precision against coverage for three confidence gates |
| `e02a_failures.py` | Failing pairs drawn over the phase; failure rate by pair length and radius |
| `e02a_certify.py` | Ridge-count and density votes on 1-wrap pairs |
| `e02a_kwrap.py` | All votes and certificates on 1-, 2- and 3-wrap pairs (the scoreboard below) |
| `e02a_heldout8.py` | Pools the k-wrap scores over PCU's eight held-out slices |
| `e03_ladders.py` | Generates certified winding ladders on one slice (`--scroll` selects the scroll) |
| `e03_sign.py` | Orients each ladder by the winding gradient of a prior fit grid |
| `e03_merge_band.py` | Merges per-slice ladders into the fitter's `relative_windings.json` |
| `e03_split_gold.py` | Holds out every n-th slice of a PCU gold band for scoring |
| `e03_fit_pod.py` | Builds the fitter dataset on a GPU machine, runs the fits, runs PCU's eval |
| `e03_figure.py` | Slips-per-wrap chart from the eval lines |

## Run

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python e01_residue_map.py --z 2250 --crop 640 --level1 --orient --out out/
```

`--z` is a level-2 slice index (band z 8500 to 9492 at full resolution is 2125 to 2373). Omit `--crop` for the full slice. Reads stream from the bucket; a 640-pixel crop takes about 20 s, nothing is cached.

## What one slice says (PHerc0826, level-2 slice 2250, full slice at level 1)

![core](figures/e01_zoom_PHerc0826_L1_z4500_full_oriented_core.png)

*Core of the scroll at level 1 (18.7 µm per pixel): wrapped phase over the CT on the left, residues on the right. Fringes track every sheet; Lasagna continues them across air gaps by design; the one cluster of residues sits on the bright debris inside the core.*

**Phase reconstruction.** The stored cosine is never zero inside the scroll; it oscillates between about 0.65 and 0.98 from sheet to sheet. Taken as is, `arccos` never reaches pi, nothing wraps, and the residue map is blind. So: contrast-normalise per local window, then take the quadrature with a Riesz transform projected on the predicted normal and use `atan2`. Pointwise sign recovery from the cosine gradient was tried first and produced balanced plus/minus residue pairs everywhere; the Riesz transform removed that. After the fix, phase rises along the predicted normal on 95% of pixels at 0.67 rad per pixel (level 2).

**What drives the residue count** (640-pixel crop, `e01_variants.py`):

| Setting | Residues per 1,000 valid pixels | Dipole share (opposite sign within 2 px) |
|---|---|---|
| level 2, normals as stored | 12.7 | 0.52 |
| level 2, normals oriented outward from the umbilicus | 11.8 | 0.51 |
| level 1 cosine, normals oriented | 2.9 | 0.24 |

Resolution matters, orientation of the normals does not. At level 2 (37 µm per pixel) packed sheets are three pixels apart and half the residues are sampling noise. Level 1 is the working resolution from here on.

**Full slice at level 1** (`figures/e01_stats_*.json`, `e01_analysis_*.json`, `e01_papyrus_split_*.json`): 13,309 residues, 10,282 after eroding the valid mask 30 px to drop the border ring, 3.8 per 1,000 cells, dipole share 0.33.

| Test | Result |
|---|---|
| Density by local fringe period | 16 per 1,000 below 3 px (59% dipoles), 4 to 6 per 1,000 at 5 to 14 px, 1.3 per 1,000 above 40 px (26% dipoles) |
| Density by distance from the umbilicus | flat, 2.3 to 4.0 per 1,000 at every radius |
| On papyrus (within 3 px of the m7 surface prediction) vs off | 3.9 vs 3.7 per 1,000 |
| Gate on Lasagna's own grad_mag (0, 0.03, 0.06) | 3.1, 3.2, 3.4 per 1,000: no effect |
| Smoothing the cosine (sigma 0.7, 1.5, 2.5 px) | 3.1, 2.9, 2.7 per 1,000: not sub-pixel noise |
| Persistence to the slice 37 µm away (same sign within 2 px) | 40% of residues, against 2.5% for a shuffled control; 59% within 4 px |

**Reading.** The residues are not noise and not a border or mask effect: they persist along z and survive smoothing. But they do not localise on anything visible either. Their density is the same on and off papyrus, flat across the scroll, and independent of Lasagna's confidence. On this slice the field has a uniform background of fringe dislocations, about four per 1,000 cells, running through z. So the residue map on its own is not a missed-sheet detector here. For unwrapping that background is what a minimum-cost-flow solver has to route branch cuts around; whether the cuts land on real sheet ambiguities is a question for scoring against the PCU held-out ladders, not for this figure.

This first experiment covered one slice and no fit; the sections below take it to a fit.

## Scored against PCU's held-out ladders (E-02a, same day)

PCU publishes 3,426 gold ladders on eight held-out slices of PHerc0826 (`gold0826_heldout8.json`, SHA256 `edc56561...`): chains of points on consecutive sheets, level-0 coordinates, z 8528 to 9424 in steps of 128. Points one rung apart are a true 1-wrap pair; two and three rungs apart give true 2- and 3-wrap pairs, so the same slices carry negatives for the +1 test. All eight slices, full width, level 1: 11,463 pairs (6,628 one-wrap, 3,202 two-wrap, 1,633 three-wrap).

Three estimates of the winding difference per pair, each independent of the others:

- **unwrap**: wrapped phase as above, unwrapped in 2D with scikit-image (reliability sorting), difference between the two points rounded to windings;
- **ridges**: interior maxima of the smoothed cosine along the segment, plus one;
- **density**: Lasagna's grad_mag (windings per base voxel, base = level 1, decoded per the manifest) integrated along the segment, times a calibration of 1.23. The calibration was fitted once, on an 800-pixel crop of the first slice (one-wrap median integral 0.81), and applied to all eight unchanged; the per-slice one-wrap median runs from 0.78 to 0.82.

| Estimate | 1-wrap accuracy | 2-wrap | 3-wrap |
|---|---|---|---|
| unwrap (on its 83 to 86% covered pairs) | 80.5% | 70.5% | 61.3% |
| ridges | 73.0% | 82.7% | 87.8% |
| density | 100% | 92.6% | 82.2% |

Certificates, pooled over all 11,463 pairs of the eight slices, any k (counts summed, not rates averaged):

| Votes that must agree | Coverage | Precision of the agreed value |
|---|---|---|
| unwrap, ridges | 51% | 98.2% |
| unwrap, density | 63% | 98.7% |
| ridges, density (no unwrap needed) | **75%** | **99.0%** (8,561 of 8,650) |
| all three | 48% | **99.7%** (5,475 of 5,490) |
| PCU's own certificate, Paris 4, 1-wrap set | 70% | 99.6% |

Per slice (`figures/e02a_heldout8_pooled.json`, one JSON per slice beside it):

| z | pairs | all three | ridges, density |
|---|---|---|---|
| 8528 | 1,411 | 99.8% at 46% | 98.7% at 75% |
| 8656 | 1,338 | 99.4% at 46% | 98.0% at 77% |
| 8784 | 1,497 | 99.3% at 50% | 98.6% at 75% |
| 8912 | 1,595 | 100% at 49% | 99.3% at 74% |
| 9040 | 1,288 | 100% at 52% | 100% at 78% |
| 9168 | 1,258 | 100% at 48% | 99.5% at 78% |
| 9296 | 1,581 | 100% at 46% | 99.8% at 75% |
| 9424 | 1,495 | 99.3% at 47% | 97.9% at 74% |

The worst slice for the three-vote certificate is 99.3%; its coverage sits between 46% and 52% everywhere. The three-vote certificate misses 15 pairs in 5,490; all 15 are 3-wrap pairs on which the three estimates agree on a wrong count. On 1- and 2-wrap pairs it has no miss in 8 slices.

What the plain unwrap gets wrong: missed wraps (0) and reversed direction (-1) on the shortest pairs; not flagged by fractional phase error, amplitude, or distance to the nearest residue (`e02a_gates.py`), and only a few points sensitive to the contrast window and smoothing. Normals are an unoriented axis (villa Lasagna docs); they are oriented outward from the umbilicus here, interpolated as a structure tensor, and a zero grad_mag is treated as an invalid sample.

![pairs](figures/e02a_failures_z8528_crop800.png)

*One-wrap gold pairs over the wrapped phase, 800-pixel crop of z 8528: green +1 (right), orange 0 (missed wrap), blue -1 (reversed), red other, grey not covered by the unwrap.*

Not done: 3D consistency across slices, a fit, slips per wrap. Reproduce with `e02a_kwrap.py` per slice (flags in its docstring: window 30, smooth 0.3, prominence 0.08, base level 1, density scale 1.23) and `e02a_heldout8.py` to pool.

## From constraints to a fit (E-03)

The certificate above scores pairs that PCU chose. A constraint source has to choose its own. `e03_ladders.py` does that on one slice: seeds on cosine ridges (one per 16 px cell at level 1), a walk outward along the oriented normal to the next cosine ridge, and each hop certified as one winding by the integrated density (within 0.35 of 1.0, a band rather than rounding, because a half-spacing hop integrates to about 0.6 and would round to 1) and by the 2D unwrap. Hops chain into ladders with relative windings, written in PCU's gold format (point collections v1, level-0 coordinates), the format the villa fitter reads as `relative_windings.json`.

Scored by overlap with the gold: a gold pair whose two rungs both lie within 2.5 px of points of one of our ladders is a test, correct if the winding difference of that ladder matches. Eight held-out slices, 34,377 ladders, 78,398 one-wrap pairs, 2,235 tests (1 to 3 wraps):

| | Correct | Precision |
|---|---|---|
| winding count, sign ignored | 2,211 | 98.9% |
| count and sign, sign from the outward normal | 2,139 | 95.7% |
| count and sign, sign from a prior fit (PCU's tracks-only fit A0, winding gradient over 120 voxels) | 2,192 | 98.1% |

Without the unwrap vote the count drops to 99.2% on one slice (spurious cosine ridges that the density band lets through), so both votes stay. The sign is the finding: where a crushed sheet doubles back, the fitter's winding decreases outward, and no local measurement can tell (the normals are an unoriented axis, the cosine is a wrapped phase). On the first slice 53 of 820 gold one-wrap pairs run inward, and every sign error of the outward rule sits on them. `e03_sign.py` therefore reads the direction of increasing winding from a prior fit grid (`grids_<run>.npz` as PCU stacks them) and reverses the ladders that run against it. A 60-voxel radius leaves 96.9%, 120 and 200 give 98.1%, 300 gives 97.5%.

The band file for the fitter is built: z 8500 to 9492 every 16 (63 slices, PCU's spacing), 268,401 ladders, 881,629 points, 3.28 points per ladder (PCU's input: 25,771 ladders, 73,110 points on the same slices). Signs: 239,686 ladders along the A0 gradient, 19,867 reversed by it, 8,848 with no readable gradient kept outward (two thirds of those on the band's first slice, where the A0 grid has no row). `relative_windings.json`, 122 MB, SHA256 `3f9c41a6...`, built by `e03_ladders.py` on each slice (about 2.7 minutes per slice on an M4), `e03_sign.py` with `grids_PHerc0826_A0.npz`, `e03_merge_band.py`; summary in `figures/e03/band_relative_windings.summary.json`. The file is not in the repository.

**The fit.** `e03_fit_pod.py` on one RTX 4090 (RunPod, 2026-10-22): villa `spiral-fitting` at PCU's commit 6bbe6e2, PHerc0826 z 8500 to 9500, tracks `surface_m7_L0_th0.2`, Lasagna level-2 normals and grad_mag, 12,000 steps, seed 1, PCU's B1w20 strip-loss weights (radius 20, dt 10, 840 draws per step), three runs that differ only in `relative_windings.json`. Scored with PCU's own `exp/e43_eval_gold.py` on the eight held-out slices (3,000 ladders, 5,788 pairs sampled with seed 0):

| Constraints | Slips per wrap | Assigned pairs | Unassigned points | Ladders satisfied by the fit |
|---|---|---|---|---|
| none (tracks only) | **49.5%** (2,042 of 4,125) | 4,125 | 1,754 | |
| PCU's `pcu_gold_0826_fit_v2.json`, 25,771 ladders | **36.1%** (1,702 of 4,714) | 4,714 | 1,210 | 46.8% |
| ours, 268,401 ladders from the Lasagna product | **35.9%** (1,733 of 4,821) | 4,821 | 1,095 | 26.9% |

![slips per wrap](figures/e03/fits/slips_per_wrap.png)

PCU published 50.1% and 36.3% for the first two rows on a T4; the reproduction here is within half a point of both, so the third row is comparable to their numbers. At PCU's settings and seed, ours matches PCU's constraint set on PCU's own scroll, with 2% more assigned pairs and 10% fewer unassigned points; the difference in slips (35.9 against 36.1) is noise. The second round below separates the two. What differs is the source: PCU's ladders are certified against a global winding field and a density model; ours come from the cosine, the normals and grad_mag of one Lasagna run, with the sign read off a tracks-only fit, and nothing else.

What the first fit did not do: it satisfied 27% of our ladders against 47% of PCU's. With ten times the ladders at the same 840 draws per step, each of ours was sampled a tenth as often. So the second round (same pod recipe, five more fits) varied the seed, the draw rate and the strip weight:

| Run | Slips per wrap | Assigned pairs | Unassigned points | Ladders satisfied |
|---|---|---|---|---|
| PCU's constraints, seed 2 | 36.8% (1,730 of 4,708) | 4,708 | 1,235 | 46.1% |
| ours, seed 2 | 34.8% (1,681 of 4,832) | 4,832 | 1,112 | 26.8% |
| ours, 8,400 draws per step | 34.4% (1,657 of 4,811) | 4,811 | 1,111 | 36.5% |
| ours, strip radius weight 40 | 33.5% (1,628 of 4,854) | 4,854 | 1,081 | 29.6% |
| ours, 8,400 draws and weight 40 | **32.1%** (1,570 of 4,890) | 4,890 | 1,029 | 40.5% |

Over two seeds at PCU's settings, ours averages 35.4% against PCU's 36.4%. Letting the fitter use the density (ten times the draws, double the radius weight) takes ours to 32.1%, four points below PCU's best, with the most assigned pairs and the fewest unassigned points of any run. The same two settings on PCU's file do not help it: weight 40 gives 36.9% (1,779 of 4,822), both settings 38.4% (1,807 of 4,707). Each set at its best: PCU 36.1%, ours 32.1%. At matched settings the two sets are within about a point of each other; the gain appears when the denser set and the higher draw rate and weight are combined, and those settings make PCU's sparser set worse.

**How sure is this?** All ten grids re-scored with PCU's own `exp/e31b_bootstrap.py` (same 3,000 ladders, seed 0; 2,000 ladder-level resamples, paired against PCU's seed-1 run):

| Run | Slips per wrap (95% interval) | Difference from PCU, seed 1, in points (95% interval) |
|---|---|---|
| tracks only | 49.5% (47.8 to 51.2) | +13.4 (+11.9 to +15.1) |
| PCU, seed 1 | 36.1% (34.5 to 37.7) | reference |
| PCU, seed 2 | 36.7% (35.2 to 38.4) | +0.6 (-0.8 to +2.0) |
| PCU, weight 40 | 36.9% (35.4 to 38.4) | +0.8 (-0.7 to +2.2) |
| PCU, 8,400 draws and weight 40 | 38.4% (36.7 to 40.1) | +2.3 (+0.9 to +3.8) |
| ours, seed 1 | 35.9% (34.4 to 37.5) | -0.2 (-1.8 to +1.5) |
| ours, seed 2 | 34.8% (33.3 to 36.4) | -1.3 (-3.0 to +0.4) |
| ours, 8,400 draws | 34.4% (33.0 to 35.8) | -1.7 (-3.3 to -0.1) |
| ours, weight 40 | 33.5% (32.1 to 35.0) | -2.6 (-4.3 to -0.9) |
| ours, 8,400 draws and weight 40 | 32.1% (30.7 to 33.6) | -4.0 (-5.7 to -2.3) |

The sampling of ladders cannot explain the best run's lead; its paired interval stays clear of zero. At PCU's settings the two sets cannot be told apart. Choosing the best of our four settings on the scoring ladders does not explain it either: picking the setting on a random half of the ladders and scoring the pick on the other half (200 splits, PCU's selection-free check) chooses ours, weight 40 8, ours, 8,400 draws and weight 40 192, and the pick scores 32.2% (range 30.7 to 34.2) against 36.1% for PCU on the same halves. Not covered by these intervals: seed-to-seed variation (about a point here; the best setting has one seed), and the generator settings (density scale 1.23, band 0.35, sign radius 120), which were tuned on the same held-out slices before any fit.

**What the scoring set can and cannot say.** PCU's held-out ladders are certified from the same Lasagna run that our constraints come from (PCU's README marks them "not human labels"). A fit pushed towards Lasagna's reading may agree better with Lasagna-derived ladders without being more correct. One number points that way: the share of independent surface tracks the fit satisfies falls as our constraints get stronger (tracks only 7.5%, PCU's set 6.9%, ours 6.0% at PCU's settings, 5.3% at the best setting). The comparison with PCU is less exposed, since PCU's own constraints come from the same generator as the scoring set, which should favour them. A check against human-traced windings (Paris 4, 1,502 labelled pairs) is the next experiment.

**On one line:** each at its best on this band, PCU's certified constraints give 36.1% slips per wrap and ours 32.1%, from a single Lasagna run with the sign taken from a tracks-only fit; whether that is fewer real winding errors needs human labels.

Pod time: about 3.5 hours including a first attempt that filled the 50 GB disk. Fits ran on 24 GB and 32 GB cards; PCU ran their 26k-ladder file on a 16 GB T4, and the 268k-ladder file at 8,400 draws per step was not tried on 16 GB. Files: `figures/e03/fits/` (eval lines, satisfaction metrics); the three fitted grids (19 MB each) are kept outside the repository: SHA256 A0 `faa40111...`, PCU `0b2a8aa3...`, ours `b3738dea...`.


## Data

- Lasagna product: `PHerc0826/representations/predictions/lasagna/20250821151701-lasagna-20260419180421/` on the open-data bucket (model 20260419180421). Channels cos, grad_mag, nx, ny; no sin or direction channels. Encodings in `lasagna_store.py`.
- CT: `PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr`.
- Umbilicus: `data/PHerc0826_umbilicus.json`, hand-annotated by us (see `data/README.md`), used only to orient normals in the generator; the fitter uses the organizers' umbilicus. This is the one human input in the PHerc0826 pipeline; the PHerc0211 run uses the organizers' file for both.

## Baseline to beat

Gill and Tran's PCU tool (https://github.com/Jashann/vesuvius-scrolling) already reads Lasagna's cosine as a wrapped phase and counts windings three ways from it (local phase cycles, an integral of wrap density, a whole-slice unwrapped field), keeping only pairs where all three agree, with the recto-surface network as a fourth vote for its gold tier: 99.62% precision at 70% coverage on Paris 4. On PHerc0826 their held-out set (`gold0826_heldout8.json`), constraint file and fit grids are published and are the scoring target here. The certificate in this repository (E-02a) is therefore a re-implementation of the same idea and mostly shows agreement with PCU's reading. What is new: a ridge-walk generator that chooses its own ladders and certifies each hop with two votes (density band and 2D unwrap), dropping PCU's global-field and recto votes; ten times the density on the same slices; the sign read from a tracks-only fit's winding gradient; and the finding that the fitter gains from that density when it draws more ladders per step at a higher strip weight, while PCU's sparser set does not.

## How this was built

By Lutfiya Miller, with Claude Code (Anthropic) as the coding and analysis agent. Every number in this README was produced by the committed scripts on the data named next to it; the per-run JSON and eval files under `figures/` are the record. Fits ran on a rented single-GPU pod with `e03_fit_pod.py`.

## License

MIT.
