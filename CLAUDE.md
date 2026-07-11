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

## Layout

```
gps_plot/
├── src/gps_plot/
│   ├── __init__.py
│   ├── plot_gps_timeseries.py        # main plot driver (console script)
│   ├── gasmatplt.py                  # GAS / matplotlib plots
│   ├── timesmatplt.py                # time-series matplotlib plots
│   ├── gmtplot.py                    # GMT-based plotting
│   ├── dev_viz.py                    # analysis-lane dev-viz (gps_analysis outputs)
│   ├── gasmatplt_workingon15May17.py # historical snapshot — DO NOT delete
│   └── gasmatplt_bgo_15May17.py      # historical snapshot — DO NOT delete
├── bin/timesmatplt-test.py           # smoke test (legacy)
├── tests/test_dev_viz.py             # dev-viz smoke tests (uv run pytest)
├── logos/                            # Veður logos used in plot output
└── pyproject.toml
```

## Dependencies

- **In**: `matplotlib`, `highlight-text`, `tornado` (runtime).
  Reads time series produced by `geo_dataread.gps_savetimes`.
  **Dev group only**: sibling `gps_analysis` (editable local path via
  `[tool.uv.sources]`) + `pytest` — powers `dev_viz.py`; production installs
  are unaffected. `uv sync` installs it.
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

## Cross-References

- `../CLAUDE.md` — ecosystem overview + dependency graph
- `../geo_dataread/CLAUDE.md` — produces the time series this package plots
- `../gps_analysis/CLAUDE.md` — the leaf math library dev_viz visualizes
- `../PLAN-analysis-lane.md` — thread C / task L5 (dev-viz), task H6 (speed pass)
- Vault hub: `/home/bgo/notes/bgovault/2.Areas/VI_GPS_Library/1776347706-gps-library-ecosystem-hub.md`

---

*Last reviewed: 2026-07-11*
