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
```

## Cleaned view (`--view cleaned`)

`timesmatplt._mask_outliers` mirrors `gps_views.read_gps_view`'s station-aware
chain (`station_step_epochs` + `resolve_protect_windows` +
`resolve_outlier_detection`), so a plotted cleaned view matches the canonical
read/write path for that station; each resolver degrades to "nothing declared"
with a `UserWarning`, never a failed plot. Declared steps are not optional — an
undeclared coseismic/equipment step over-flags or trips the excess-candidate
abort at ANY threshold. Flags MASK (NaN) + grey overlay.

Three marker states, and the distinction is what each COMMITS to:

| marker | meaning | in the series? |
|---|---|---|
| red | clean | yes |
| grey | flagged outlier | **no** (NaN) |
| **gold** | **provisional** — recent, indeterminate step evidence | **yes** |

Gold epochs are the ones the detector cannot rule on yet (no data follows
them, so a blunder and the onset of deformation look identical). They stay in
the data; the marker only says the verdict is pending, and it WILL change as
epochs arrive. `--hide-outliers` drops the grey overlay (display only — the
epochs are already masked; the y-axis then tightens to the cleaned series,
often dramatically: RHOF north 90 mm → 45 mm of range) but deliberately does
NOT hide gold: decluttering removes decided outliers, never undecided ones.
`--provisional-days` bounds the recency window (0 disables; default 14 from
`geo_dataread`), and the bound matters — indeterminate clusters also sit at
old mid-series gaps and would otherwise dominate.

```bash
plot-gps-timeseries RHOF --view cleaned --outlier-param window_n_sigma=3.0
plot-gps-timeseries RHOF --view cleaned --outlier-overrides ./overrides.csv
plot-gps-timeseries RHOF --view cleaned --uncert 10 --hide-outliers
```

`--uncert` is the first lever for "obvious outliers survive": detection
whitens by the formal σ, so a large excursion with a large error bar is not
anomalous (RHOF: +83 mm Up at σ=13.2 mm scores only 3.3σ). The default 15 mm
lets those through; `--uncert 10` drops 40 of 4713 epochs and removes most of
them. Without σ at all the flag count rises ~45 %.

`--outlier-param NAME=VALUE` (repeatable, also `plotTime(outlier_params=)`)
builds a `gps_analysis.OutlierParams` off the dataclass itself — no default is
restated here, and an unknown NAME lists the valid ones. Any override REPLACES
the station's `outlier_overrides.csv` row; unset defers to that catalog, else
spec defaults. `min_outlier=` is the scalar floor, NOT the catalog's
per-component `[N,E,U]` vector.

Loosening is **not monotonic** — RHOF 2023→ (3483 component-epochs):
`window_n_sigma` 4.0 (spec) → 20 flagged, 3.0 → 55, 2.0 → **0**, because at k=2
the candidate fraction passes `max_flag_fraction=0.05` and the excess-candidate
rule aborts, silently serving raw. A sudden drop to zero means abort, not a
clean station. PLAN Phase 3: consume `read_gps_view`'s `{comp}_cleaned` /
`{comp}_outlier` instead of re-deriving here.

## Detrend workbench (`gps-detrend-workbench`)

One station, one PDF: plate-frame series + fitted trajectory, and the detrended
series. Detrend choice is **curation, not computation** — which window is
pre-unrest, whether a station supports periodic terms, whose parameters to
borrow. All estimation is called from `geo_dataread.detrend_estimate`
(`station_record_from_arrays`, `resolve_fit_settings`, `build_document`), so
workbench and batch `gps-estimate-detrend` can never disagree about what a
record means. Detection runs the FULL pipeline, falling back loudly to S0-only
if the excess-candidate rule aborts — an S0 record leaves model-visible outliers
in the fit, so on a fallback declaring the missing step beats accepting it
(measured on RHOF vertical: 0.54 mm/yr of rate difference).

**First `gps_plot` module that writes config.** `--commit` merge-writes one
station into `detrend_params.json`, preserving the rest; without it nothing is
stored. Proven end-to-end: `plot-gps-timeseries <STA> --view detrended` then
renders from that record.

```bash
# catalogs deployed to ~/.config/gpsconfig 2026-07-29 — no GPS_CONFIG_PATH needed
gps-detrend-workbench SELF --max-gap-years 2.0 --out SELF-iter1.pdf
gps-detrend-workbench RHOF --model periodic --donor VMEY --commit
```

`--out` shares the scratch figdir with `tools/local-plot/figview.sh`: a bare
filename lands in `$FIGDIR`, else the checkout's gitignored `tmp-figdir/`, else
CWD. A path with a separator is honoured verbatim, so an exported `FIGDIR` can
never relocate an explicit `--out`. `--max-gap-years` is effectively required —
the 0.5 default rejects every station in the working set.

Events are **declared, never detected** (tier A), in three colours: `darkgreen`
TOS equipment changes (live `tostools`; one row per DEVICE join, coalesced to
distinct days), `darkred` seismic (`steps.csv` rows whose `kind` is
earthquake/coseismic/seismic, plus `--event YYYYMMDD[,LABEL]`), `royalblue` the
fit. Seismic lines read `steps.csv` via `gps_parser.outlier_catalogs.read_steps`,
**not** `gps_views.station_step_epochs` — that one drops `kind`, so it cannot
separate an earthquake from an antenna swap. No skjálftalísa client exists
anywhere in the ecosystem (planned only, in `analysis.yaml` + the CSV header).

Offsets are declare-and-fit: epochs FIXED, amplitudes estimated and shown at
once. Epoch detection is absent deliberately — it is *circular* today: a jump
detector needs clean data, the outlier detector needs declared jumps to make it
(SELF: 9.1 % candidates and abort until one step was declared).

Gaps, measured: `--terms` does NOT round-trip (`model=` is stored at fit time,
`terms=` is per-call and unstored — two different decisions, 16.48 mm max
divergence); `--donor` copies rather than points, so it will not follow a
re-estimated donor; workbench/batch `uncert` defaults differ (10 vs 15) and
`uncert` is absent from `refs`; the `max_gap_years=0.5` gate fails every station
in the working set, so pass `--max-gap-years`.

Fractional-year epochs are at **noon** — 2008-05-29 is `149.5/366 = 2008.40847`,
not `149/366`. That trap and the TOT join live in `tools/local-plot/README.md`.

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
- `tools/local-plot/README.md` — local figure workflow, TOT join, epoch/shell traps
- `.interrogate-detrend-workbench.md` — workbench destination doc (gitignored)
- `../PLAN-analysis-lane.md` — thread C / task L5 (dev-viz), task H6 (speed pass)
- Vault hub: `/home/bgo/notes/bgovault/2.Areas/VI_GPS_Library/1776347706-gps-library-ecosystem-hub.md`

---

*Last reviewed: 2026-07-29 (detrend workbench T0–T6: curation levers, --commit
merge-write + round-trip proof, TOS + seismic event lines; earlier — cleaned view: station-aware resolution chain +
`--outlier-param` / `--outlier-overrides`; map lane deformation slices:
velocity_map / deformation_vectors / slip_map + dem_grid, optional `maps` extra,
ruff scope for the modern lane)*
