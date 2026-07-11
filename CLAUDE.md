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

## Map lane (PyGMT, PLAN Phase 3 — slice 1)

`maps.py::station_map(stations, *, region=None, projection="M12c", ...)` —
GSHHG coastline base map + station markers + labels. Coordinates come from
`stations.cfg` via `gps_parser` (`station_coordinates()`, same access path as
`gps_api.precompute`); pre-resolved `StationCoordinate` objects can be passed
instead (no config needed). Region defaults to the station bounding box +
margin. Zero hardcoding: style/region/output are parameters (`DEFAULT_*`
constants). Returns the `pygmt.Figure`; `outfile=` saves (format from suffix).

```python
from gps_plot.maps import station_map
station_map(["RHOF", "AKUR"], title="North Iceland", outfile="stations.png")
```

Next slices (seams documented in the module docstring): slice 2 =
`velocity_map` (adds `fig.velo` layers in the GMT velo schema from
`gps_analysis` velocity output, on top of `station_map`'s figure); slice 3 =
DEM/hillshade background. The dormant `gmtplot.py::velomap` is the recipe
reference for both until absorbed (note: its `fig.text` had lon/lat swapped —
fixed in `maps.py`).

## Cross-References

- `../CLAUDE.md` — ecosystem overview + dependency graph
- `../geo_dataread/CLAUDE.md` — produces the time series this package plots
- `../gps_analysis/CLAUDE.md` — the leaf math library dev_viz visualizes
- `../PLAN-analysis-lane.md` — thread C / task L5 (dev-viz), task H6 (speed pass)
- Vault hub: `/home/bgo/notes/bgovault/2.Areas/VI_GPS_Library/1776347706-gps-library-ecosystem-hub.md`

---

*Last reviewed: 2026-07-11 (map lane slice 1: `maps.py::station_map`, optional
`maps` extra, ruff scope for the modern lane)*
