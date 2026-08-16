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
New (2026-07): `detrend_workbench.py` — operator workbench for detrend curation
(T0–T6 complete, DoD 1–5 met); see Detrend workbench below.
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
│   ├── detrend_workbench.py          # operator workbench — the only config-WRITING module
│   ├── gasmatplt_workingon15May17.py # historical snapshot — DO NOT delete
│   └── gasmatplt_bgo_15May17.py      # historical snapshot — DO NOT delete
├── bin/timesmatplt-test.py           # smoke test (legacy)
├── tests/test_dev_viz.py             # dev-viz smoke tests (uv run pytest)
├── tests/test_maps.py                # map-lane tests (render test env-gated on GMT)
├── tests/test_detrend_workbench.py   # round-trip, safety rails, TOS (tos_SELF.json fixture)
├── tools/local-plot/                 # local figure workflow + TOT join procedure (README)
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
gps-detrend-workbench ... # entry: gps_plot.detrend_workbench:main
gps-detrend-picker-qt ... # entry: gps_plot.detrend_picker_qt:main (pyqtgraph)
gps-detrend-picker ...    # entry: gps_plot.detrend_picker:main (marimo notebook)
```

## Detrend lane → `docs/detrend-lane.md` 📄

The cleaned view, the detrend workbench, segments and the Qt picker live in
**[`docs/detrend-lane.md`](docs/detrend-lane.md)** — 300 lines of measured
behaviour that belong one level down. Read it before touching any of them.

- **Cleaned view** (`--view cleaned`) — three marker states, and what each
  COMMITS to: red clean, grey flagged (NaN, removed), **gold provisional**
  (kept — recent, indeterminate, verdict pending). Loosening `window_n_sigma`
  is NOT monotonic: a sudden drop to zero flagged means the excess-candidate
  rule ABORTED and you are being served raw data.
- **Detrend workbench** (`gps-detrend-workbench`) — detrend choice is
  curation, not computation. The only config-WRITING module in the package.
- **Segments** (`--segment A:B`, repeatable) — the fit domain is a union of
  intervals, which answers four asks at once; the gates changed meaning per
  segment.
- **Qt picker** (`gps-detrend-picker-qt`) — layered ON TOP of the CLI, and its
  whole promise is one invariant: **the emitted command reproduces the
  figure.** Four violations of it are catalogued there, all the same shape —
  a second place assembling the same decision.

**Per-station curation is config, not just a record.** `--commit` writes three
things a batch re-run must find, all merge-written one station at a time into
the deployed files: the record → `detrend_params.json`; the stage plan →
`analysis.yaml` `detrend.estimation.stage_plans`; the model + `--term`
transients → `analysis.yaml` `detrend.estimation.models`. A fit-time decision
living only inside the record is invisible to `gps-estimate-detrend`, which
RECOMPUTES the record — that was true of stage plans until 2026-08-16 and of
model/terms until the same day.

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
- **[`docs/detrend-lane.md`](docs/detrend-lane.md)** 📄 — cleaned view, workbench,
  segments, Qt picker (the deep lane; read before touching any of them)
- `tools/local-plot/README.md` — local figure workflow, TOT join, epoch/shell traps
- `.interrogate-detrend-workbench.md` — workbench destination doc (gitignored)
- `../PLAN-analysis-lane.md` — thread C / task L5 (dev-viz), task H6 (speed pass)
- Vault hub: `/home/bgo/notes/bgovault/2.Areas/VI_GPS_Library/1776347706-gps-library-ecosystem-hub.md`

---

*Last reviewed: 2026-08-16 (detrend lane routed out to `docs/detrend-lane.md` — this file was 453 lines against the 150-line subdir budget; moved VERBATIM, since every measured number in it was expensive to obtain. `--commit` now stores the model + `--term` transients to `analysis.yaml` `detrend.estimation.models` as well as the stage plan, because the batch RECOMPUTES the record and a fit-time decision living only inside it cannot be found; earlier — Qt picker section added — the emitted command reproduces the figure, and the four ways that broke; `_mask_outliers` returns FOUR members (the fourth, `aborted`, is what makes an aborted axis distinguishable from a clean one) — the annotation said three; the equipment-line count in `tos_equipment_epochs` was stale at 6 → 5 from before `MIN_DEPLOYMENT_DAYS`, verified 6 → 2 against the fixture; earlier — workbench `--show-outliers`: inverted emphasis —
flagged red, the rest grey — display-only, exclusive with `--hide-outliers`,
lanes still countable; makes an aborted component legible as an all-grey axis
(NYLA); earlier — workbench out-of-window screen: the view detector
fills the fit's silence outside the window as a second, hollow-grey lane with
its own count — record untouched; `timesmatplt.view_flags` is now the single
detector call site (`restrict=`/`step_epochs=`); earlier — round-trip fidelity: --commit refused
under a non-default --terms, `uncert` expressible + recorded on both the
workbench and batch sides; earlier — workbench grey outlier overlay + --hide-outliers,
fed by the fit's own inlier mask via geo_dataread station_estimate_from_arrays;
earlier — detrend workbench T0–T6: curation levers, --commit
merge-write + round-trip proof, TOS + seismic event lines; earlier — cleaned view: station-aware resolution chain +
`--outlier-param` / `--outlier-overrides`; map lane deformation slices:
velocity_map / deformation_vectors / slip_map + dem_grid, optional `maps` extra,
ruff scope for the modern lane)*
