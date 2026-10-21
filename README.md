# winding-phase

**Can Lasagna's cosine channel be treated as a wrapped phase and unwrapped, InSAR style, into winding constraints for the villa spiral fitter?**

Status, 2026-10-21: one slice of PHerc0826, full width at level 1. Phase reconstruction works and tracks the sheets; the residues (fringe dislocations) form a uniform, z-persistent background rather than marking merges or tears. No constraints exported yet, nothing scored against a fit.

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

Not done: no unwrapping, no constraints exported, no comparison with a fit. One slice of one scroll.

## Data

- Lasagna product: `PHerc0826/representations/predictions/lasagna/20250821151701-lasagna-20260419180421/` on the open-data bucket (model 20260419180421). Channels cos, grad_mag, nx, ny; no sin or direction channels. Encodings in `lasagna_store.py`.
- CT: `PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr`.
- Umbilicus: `data/PHerc0826_umbilicus.json`, ours, see `data/README.md`.

## Baseline to beat

Gill and Tran's PCU tool (https://github.com/Jashann/vesuvius-scrolling): certified relative-winding constraints, 99.62% precision at 70% coverage on Paris 4; on PHerc0826 their held-out set (`gold0826_heldout8.json`) and fit grids are published and are the scoring target here.

## License

MIT.
