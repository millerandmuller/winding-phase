# data/

- `PHerc0826_umbilicus.json`: hand-annotated umbilicus polyline for PHerc0826
  (volume 20250821151701, level-0 coordinates), our own, verified during the
  August 2026 entry (millerandmuller/first-light-pherc0826). MIT.

- `PHerc0211_umbilicus.json`: the organizers' umbilicus for PHerc0211 (volume
  20250821151803), copied unchanged from the open-data bucket
  (`PHerc0211/representations/umbilicus/20250821151803-umbilicus-20260808112626.json`,
  2026-10-23). Same format: control points in level-0 coordinates.

- `PHercParis4_umbilicus.json`: the umbilicus from the organizers' Paris 4
  spiral dataset (`https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/umbilicus.json`,
  2026-10-24), copied unchanged. Control points in the dataset's 9.6 um frame
  (level 2 of the 2.4 um scan), which is the frame the generator writes for
  Paris 4. Paris 4 is used only for the human-label check.
