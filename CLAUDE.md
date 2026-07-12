# gps_plot

Time-series visualization for GPS displacement data and auxiliary geophysical signals.
Legacy maintenance — used for research plots and dashboard exports — plus the new
**analysis-lane dev-viz** (`dev_viz.py`, PLAN-analysis-lane §4 L5).

## Status

Legacy modules: maintenance only, ~3362 LOC. Two dated source snapshots are kept as
historical reference (do not delete without coordinating).
New (2026-07): `dev_viz.py` — development visualization of `gps_analysis` outputs
(trajectory fit, detrended residuals, GBIS4TS break points, WLS velocity). Modern
code: typed, mypy-strict/ruff/black clean, tested (`tests/test_dev_viz.py`).
New (2026-07): `maps.py` — PyGMT map lane slice 1 (`station_map`); see Map lane below.
Ruff lint/format scope covers the modern lane only — legacy modules and the
dated snapshots are excluded in `[tool.ruff]` (`pyproject.toml`).

## Layout

```
gps_plot/
├── src/gps_plot/
│   ├── __init__.py
│   ├── plot_gps_timeseries.py        # main plot driver (console script)
│   ├── gasmatplt.py                  # GAS / matplotlib plots
│   ├── timesmatplt.py                # time-series matplotlib plots
│   ├── gmtplot.py                    # dormant velomap seed (recipe ref for maps slice 2/3)
│   ├── maps.py                       # PyGMT map lane: station_map (optional 'maps' extra)
│   ├── dev_viz.py                    # analysis-lane dev-viz (gps_analysis outputs)
│   ├── gasmatplt_workingon15May17.py # historical snapshot — DO NOT delete
│   └── gasmatplt_bgo_15May17.py      # historical snapshot — DO NOT delete
├── bin/timesmatplt-test.py           # smoke test (legacy)
├── tests/test_dev_viz.py             # dev-viz smoke tests (uv run pytest)
├── tests/test_maps.py                # map-lane tests (render test env-gated on GMT)
├── logos/                            # Veður logos used in plot output
└── pyproject.toml
```

## Dependencies

- **In**: `matplotlib`, `highlight-text`, `tornado` (runtime).
  Reads time series produced by `geo_dataread.gps_savetimes`.
  **Optional `maps` extra**: `pygmt>=0.19` (PyPI) on top of a **system GMT ≥ 6**
  C library (OS package, conda-forge `gmt`, or from-source via
  `GMT_LIBRARY_PATH`) — `uv sync --extra maps`. Import of `gps_plot`/the
  matplotlib lane never needs it (lazy guard in `maps.py`).
  **Dev group only**: siblings `gps_analysis` + `gps_parser` (editable local
  paths via `[tool.uv.sources]`), `pytest`, `ruff` — power `dev_viz.py` and
  the map-lane tests; production installs are unaffected. `uv sync` installs it.
- **Out**: end-user plots. No internal package consumes this.

## Console Scripts

```bash
gps_plot                  # entry: gps_plot:main
plot-gps-timeseries ...   # entry: gps_plot.plot_gps_timeseries:main
gps-analysis-devviz ...   # entry: gps_plot.dev_viz:main (dev group required)
```

## Dev-viz (analysis lane, thread C / L5)

Three shared-axis panels for one station/component: observed + `lineperiodic`
fit with WLS velocity annotation (rate ± σ, horizontal |v|/azimuth ± σ);
detrended residuals; GBIS4TS BPD1/BPD2 optimal break model on the break-input
series with break epochs marked. All math is called from `gps_analysis` —
nothing is re-derived here. `--break-input` selects what detection runs on:
`raw` (zero-referenced observed, default — break epoch recovers, rates absorb
seasonal on short windows), `seasonal_removed`, or `residuals` (known-bad:
the lineperiodic fit absorbs the ramp — kept to keep the failure visible).

```bash
uv run gps-analysis-devviz --synthetic --component east --out /tmp/devviz.png
uv run gps-analysis-devviz --neu STA.NEU --component north --breaks 2 --out sta.png
uv run gps-analysis-devviz --synthetic --no-detect ...   # skip MCMC, overlay truth
```

Data source is a parameter: `load_neu()` reads published `.NEU` products
(`date time dN DN dE DE dU DU`, mm — aflogun-verified format);
`synthetic_station()` generates BPD+seasonal+noise series so the viz runs
without station data. MCMC dev default is 2 000 kept iterations on a 1-yr
series (`--runs`/`--t-runs`/`--days`; production GBIS4TS uses 1e6 — the dev
posterior is indicative only). Pre-H3 cost warning: each noise-parameter
step factorizes the dense N×N covariance (~10 ms at N=365, ~0.3 s at N=730),
so long series make detection slow until the Toeplitz/Cholesky speedup lands.

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

## Cross-References

- `../CLAUDE.md` — ecosystem overview + dependency graph
- `../geo_dataread/CLAUDE.md` — produces the time series this package plots
- `../gps_analysis/CLAUDE.md` — the leaf math library dev_viz visualizes
- `../PLAN-analysis-lane.md` — thread C / task L5 (dev-viz), task H6 (speed pass)
- Vault hub: `/home/bgo/notes/bgovault/2.Areas/VI_GPS_Library/1776347706-gps-library-ecosystem-hub.md`

---

*Last reviewed: 2026-07-12 (map lane deformation slices: velocity_map / deformation_vectors / slip_map + dem_grid, optional
`maps` extra, ruff scope for the modern lane)*
