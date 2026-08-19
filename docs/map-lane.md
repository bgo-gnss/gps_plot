# gps_plot — the map lane (PyGMT)

Deep context for `maps.py`, routed out of `gps_plot/CLAUDE.md` when that file
passed its 150-line subdir budget a second time. **Moved verbatim** — the
conventions below are what keep four map functions from each inventing their
own coordinate source and colour vocabulary.

Read `../CLAUDE.md` first for the package summary and cross-references.

---

## Map lane (PyGMT, PLAN Phase 3 — deformation lane live)

`maps.py` — four map functions sharing the slice-1 conventions (coords from
`stations.cfg` via `gps_parser` or pre-resolved `StationCoordinate`s, zero
hardcoding — `DEFAULT_*` params, lazy pygmt guard, returns `pygmt.Figure`,
`outfile=` saves). All take `dem_grid=` for a hillshade background (slice 3).

- `station_map(stations, ...)` — coastline base + station markers/labels.
- `velocity_map(vectors, ...)` — `fig.velo` arrows + 1-σ error ellipses.
  Input: `VelocityVector` records from `velocity_vectors(stations, e, n, σe,
  σn)` (arrays, mm/yr) or `velocity_vectors_from_geojson()` (the `gps_api`
  `GET /velocities` product). WLS formal σ vs MLE honest σ flow through as-is
  (bigger ellipses); mixed `method` tags become color-coded layers
  (`DEFAULT_METHOD_COLORS`) + legend.
- `deformation_vectors(stations, obs_e, obs_n, models, ...)` — observed
  displacement field (mm, NaN = missing) vs N model `VectorField`s, with
  source stars, scale-reference arrow + color-keyed legend. Model fields via
  `mogi_model_field()` / `okada_model_field()` (lazy `gps_analysis` forwards;
  Mogi params match the `gps_api` `DeformationResult` product) or pass
  pre-computed mm arrays. `examples/mogi_vector_comparison.py` (Svartsengi
  observed vs two Mogi models + sample PNG) is now a thin caller of this.
- `slip_map(slip, ...)` — Okada distributed slip as colored fault patches +
  colorbar. Consumes the `gps_api` `SlipDistributionResult` mapping
  (`models/<region>_slip.json`; corners from per-patch lon/lat + plane
  geometry via `slip_patches_from_product()`, metric from
  `gps_analysis.local_coordinates`) or pre-built `SlipPatch` polygons.
  `view="map"` (surface projection) or `view="plane"` (along-strike ×
  down-dip km cross-section — the readable view for near-vertical dikes).

```python
from gps_plot.maps import station_map
station_map(["RHOF", "AKUR"], title="North Iceland", outfile="stations.png")
```

The dormant `gmtplot.py::velomap` recipe is now fully absorbed (velo layers,
DEM hillshade; its `fig.text` lon/lat swap fixed). Render tests are env-gated
on pygmt/GMT (`GMT_LIBRARY_PATH=$HOME/git/gmt/install/lib uv run pytest`).

## Installing the lane

The `maps` extra is `pygmt>=0.19` (PyPI) on top of a **system GMT ≥ 6** C
library (OS package, conda-forge `gmt`, or from-source via
`GMT_LIBRARY_PATH`) — `uv sync --extra maps`. Import of `gps_plot` or the
matplotlib lane never needs it: `maps.py` guards the pygmt import lazily, so
a production install without GMT is unaffected.

## Cross-References

- `../CLAUDE.md` — package summary, layout, console scripts
- `../../gps_analysis/CLAUDE.md` — Mogi/Okada forward models behind the
  model fields
- `../../gps_api/CLAUDE.md` — the `/velocities` and slip-distribution
  products these consume
- `detrend-lane.md` — the other deep lane (cleaned view, workbench, pickers)
