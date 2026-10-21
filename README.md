# winding-phase

**Can Lasagna's cosine channel be treated as a wrapped phase and unwrapped, InSAR style, into winding constraints for the villa spiral fitter?**

Status, 2026-10-21: first look at one slice of PHerc0826. Phase reconstruction works; residues (fringe dislocations) are counted and drawn over the CT. No constraints exported yet, nothing scored against a fit.

## What is here

| File | Does |
|---|---|
| `lasagna_store.py` | Reads the published Lasagna pyramids (cos, grad_mag, nx, ny) and the CT for one scroll from the open-data bucket, one slice window at a time; decodes the uint8 encodings |
| `phase.py` | Wrapped phase per slice: local contrast normalisation, Riesz-transform quadrature projected on the predicted normal, `atan2`; 2x2 loop residues |
| `umbilicus.py` | Outward reference direction from the hand-annotated umbilicus, used to orient the normals |
| `e01_residue_map.py` | One slice: three-panel figure (CT, wrapped phase, residues over CT) plus a JSON of the numbers |
| `e01_variants.py` | Same crop under three settings, to see what drives the residue count |

## Run

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python e01_residue_map.py --z 2250 --crop 640 --level1 --orient --out out/
```

`--z` is a level-2 slice index (band z 8500 to 9492 at full resolution is 2125 to 2373). Omit `--crop` for the full slice. Reads stream from the bucket; a 640-pixel crop takes about 20 s, nothing is cached.

## What one slice says so far (z 2250, 640-pixel crop around the umbilicus)

The stored cosine is never zero inside the scroll; it oscillates between about 0.65 and 0.98 from sheet to sheet. Taken as is, `arccos` never reaches pi and no phase wraps, so the phase has to be contrast-normalised per local window first. Pointwise sign recovery from the cosine gradient was too noisy (balanced plus/minus residue pairs everywhere); the Riesz transform fixes that.

| Setting | Residues per 1,000 valid pixels | Share that are dipoles within 2 px |
|---|---|---|
| level 2, normals as stored | 12.7 | 0.52 |
| level 2, normals oriented outward | 11.8 | 0.51 |
| level 1 cosine, normals oriented | 2.9 | 0.24 |

Orientation of the normals barely matters; resolution does. At level 2 (37 µm per pixel) tightly packed sheets are three pixels apart and half the residues are sampling noise. Level 1 is the working resolution.

What the figure shows at level 1: extended residue lines where sheets fold back on themselves, and a dense residue carpet in the packed outer wraps and in the compressed region. Whether that carpet is real sheet topology or still noise is the next question.

## Data

- Lasagna product: `PHerc0826/representations/predictions/lasagna/20250821151701-lasagna-20260419180421/` on the open-data bucket (model 20260419180421). Channels cos, grad_mag, nx, ny; no sin or direction channels. Encodings in `lasagna_store.py`.
- CT: `PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr`.
- Umbilicus: `data/PHerc0826_umbilicus.json`, ours, see `data/README.md`.

## Baseline to beat

Gill and Tran's PCU tool (https://github.com/Jashann/vesuvius-scrolling): certified relative-winding constraints, 99.62% precision at 70% coverage on Paris 4; on PHerc0826 their held-out set (`gold0826_heldout8.json`) and fit grids are published and are the scoring target here.

## License

MIT.
