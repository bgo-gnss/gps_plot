# gps_plot

Time-series visualization for GPS displacement data and auxiliary geophysical signals.
Legacy maintenance — used for research plots and dashboard exports — plus the new
**analysis-lane dev-viz** (`dev_viz.py`, PLAN-analysis-lane §4 L5).

## Status

Legacy modules: maintenance only, ~3362 LOC; two dated snapshots are kept as
historical reference (do not delete without coordinating). Modern lane
(2026-07→): `dev_viz.py`, `maps.py`, `detrend_workbench.py` + the two pickers,
each routed to its own doc below.

**Quality gates.** `uv run ruff check` / `ruff format --check` / `mypy src`
are CI-enforced; `uv run pytest` is local-only (reads real station data).
mypy is `strict`, scoped to the modern lane like the ruff excludes (legacy
modules `ignore_errors`, `tests/` out). Scope rationale: `docs/mypy-status.md`.

## Layout

```
gps_plot/
├── src/gps_plot/
│   ├── cli/                          # gpsplot: dispatcher (ts, stack, seasonal, rtk, map, devviz, workbench, picker); common.py = shared options
│   ├── stack.py                      # gpsplot stack: several stations/baselines, patterns ("SEY?-SEYD")
│   ├── seasonal.py                   # gpsplot seasonal: seasonal folded on one year — data, own, regional LOO
│   ├── compose.py                    # multi-series figure (stack_figure, label-keyed colours)
│   ├── axes.py                       # baseline series on other axes (orient: values/σ/overlays/ends); pair_axes/sign_pair = N/E/U, + = lengthening (sources.gps.fetch signs daily pairs; Orienter for RTK); label_flags
│   ├── direction.py                  # gpsplot direction: motion direction (gps_analysis.direction) of pairs/stations → a station_areas.yaml directions: line
│   ├── polar.py                      # gpsplot polar: sector / terrain map / section of the horizontal displacement since an onset, --together; A_B-polar-<period>[-rtk]
│   ├── terrain.py                    # DEM slope at a station (plane over 50 m; 2 m ArcticDEM area grid where it covers, else 5 m ISN93)
│   ├── labels.py                     # is/en label table (components, '+ fjær'); axis_labels
│   ├── rtk.py                        # gpsplot rtk: RTK baselines (geo_dataread.rtk_read), --daily GLOBK join; --sign/--axes (docs/baselines.md)
│   ├── sources/rtk.py                # RTK baseline -> Series (mm, zeroed on the first hour); align_daily
│   ├── map.py                        # gpsplot map: station/velocity/deformation/slip over maps.py
│   ├── sources/velocity.py           # per-station windowed velocity / displacement / stored secular rate
│   ├── plot_gps_timeseries.py        # time-series driver: build_parser + run (gpsplot ts)
│   ├── series.py                     # Series: what every source returns, plots draw
│   ├── sources/gps.py                # station / baseline -> Series (fetch); model line (fetch_model)
│   ├── layers/model.py               # the stored-model line (gpsplot ts --model)
│   ├── trajectory.py                 # model sampled for drawing (dense grid, steps bracketed)
│   ├── timesmatplt.py                # time-series matplotlib plots
│   ├── gasmatplt.py                  # GAS plots (legacy, ruff-excluded)
│   ├── gmtplot.py                    # dormant velomap seed — absorbed into maps.py
│   ├── maps.py                       # PyGMT map lane (optional 'maps' extra)
│   ├── dev_viz.py                    # analysis-lane dev-viz (gps_analysis outputs)
│   ├── detrend_workbench.py          # operator workbench — the only config-WRITING module
│   ├── detrend_picker_qt.py          # Qt picker — the one GUI; stores in-process
│   └── gasmatplt_{bgo,workingon}15May17.py  # Py-2 snapshots — DO NOT delete
├── docs/                             # gpsplot-commands, baselines, detrend-lane, map-lane, dev-viz, plots-yaml, …
├── tests/                            # dev_viz, maps (GMT-gated), workbench, pickers
├── bin/timesmatplt-test.py           # smoke test (legacy)
├── tools/local-plot/                 # local figure workflow + TOT join procedure (README)
├── logos/                            # Veður logos used in plot output
└── pyproject.toml
```

## Dependencies

- **In**: `matplotlib`, `highlight-text`, `tornado` (runtime).
  Reads time series produced by `geo_dataread.gps_savetimes`.
  **Optional `maps` extra**: `pygmt` + a system GMT ≥ 6 C library — see
  `docs/map-lane.md`. The matplotlib lane never needs it.
  **Dev group only**: siblings `gps_analysis` + `gps_parser` (editable local
  paths via `[tool.uv.sources]`), `pytest`, `ruff` — power `dev_viz.py` and
  the map-lane tests; production installs are unaffected. `uv sync` installs it.
- **Out**: end-user plots. No internal package consumes this.

## Console Scripts

```bash
gpsplot [ts] STA ...      # entry: gps_plot.cli:main (ts is the default subcommand)
gps_plot ...              # same as gpsplot
plot-gps-timeseries ...   # old name of gpsplot ts, kept until cron/scripts move
gps-analysis-devviz ...   # entry: gps_plot.dev_viz:main (dev group required)
gps-detrend-workbench ... # entry: gps_plot.detrend_workbench:main
gps-detrend-picker-qt ... # entry: gps_plot.detrend_picker_qt:main (pyqtgraph)
```

## gpsplot subcommands → `docs/gpsplot-commands.md` 📄

One command, `gpsplot`, one subcommand per plot kind (`cli/__init__.py`
`COMMANDS`; only the chosen one is imported): ts, stack, seasonal, rtk,
rtk-run, map, direction, polar, run, devviz, workbench, picker. `ts` options
live once in `plot_gps_timeseries.build_parser`. Production: `gpsplot ts
active --strict --manifest FILE`, `gpsplot run` over plots.yaml. Every option
and behaviour (views, outliers, icing/snow/network days, common mode, rates,
batch contract, maps, tracks, layers): **[`docs/gpsplot-commands.md`](docs/gpsplot-commands.md)**;
design + staged plan: `docs/cli-subcommands-design.md`. Every step is gated by
`bin/check_cli_variants.py` (byte-identical PNGs vs a `main` export).

## Detrend lane → `docs/detrend-lane.md` 📄

The cleaned view, the detrend workbench, segments and both pickers: read
**[`docs/detrend-lane.md`](docs/detrend-lane.md)** before touching any of them.
Two things to know: zero flagged points in the cleaned view = the
excess-candidate rule ABORTED and you are seeing raw data; and
`detrend_workbench.py` is the only config-WRITING module (record +
`steps.yaml` + `analysis.yaml` keys — a batch re-run must find all three).

## Baselines → `docs/baselines.md` 📄

A-B = rover − reference; + = lengthening per N/E/U (`--sign apart`, default;
`raw` = old A − B); `--axes neu|along|3d|motion|slope` on ts/stack/rtk;
station_areas.yaml `directions:`; `gpsplot direction`, `gpsplot polar`, the
2 m DEM grids, and the daily baseline layering (`geo_dataread.baseline`).

## Dev-viz → `docs/dev-viz.md` 📄

`dev_viz.py` (`gps-analysis-devviz`): observed + fit, residuals and break
model for one station/component, all math from `gps_analysis`. `--break-input`
has a known-bad value on purpose, and the dev MCMC (2 000 iterations) is
indicative only — both in the doc.

## Map lane → `docs/map-lane.md` 📄

`maps.py` — four PyGMT functions (`station_map`, `velocity_map`,
`deformation_vectors`, `slip_map`) over `gps_api` products, coords from
stations.cfg, `dem_grid=` hillshade; `gpsplot map` (station/velocity/
deformation/slip, `--tos`, `--dem iceland`, regional layers and insets).
Signatures, inputs and the GMT install story are in the doc; config guide:
`docs/map-config-guide.md`.

## Cross-References

- `../CLAUDE.md` — ecosystem overview + dependency graph
- `../geo_dataread/CLAUDE.md` — produces the time series this package plots
- `../gps_analysis/CLAUDE.md` — the leaf math library dev_viz visualizes
- **[`docs/detrend-lane.md`](docs/detrend-lane.md)** 📄 — cleaned view, workbench,
  segments, Qt picker (the deep lane; read before touching any of them)
- **[`docs/map-lane.md`](docs/map-lane.md)** 📄 — the four PyGMT map functions,
  their `gps_api` inputs, and the GMT install story
- **[`docs/gpsplot-commands.md`](docs/gpsplot-commands.md)** 📄 — every gpsplot
  option and behaviour (ts, stack, maps, tracks, layers, batch contract)
- **[`docs/baselines.md`](docs/baselines.md)** 📄 — baseline sign, rotated axes,
  directions, `gpsplot direction`/`polar`, terrain DEMs, plots.yaml runs
- **[`docs/dev-viz.md`](docs/dev-viz.md)** 📄 — `--break-input`, MCMC dev defaults
- **[`docs/mypy-status.md`](docs/mypy-status.md)** 📄 — the typing gate's scope,
  the 137 → 0 path, and the (now cleared) `gps_analysis` blocker
- `tools/local-plot/README.md` — local figure workflow, TOT join, epoch/shell traps
- `.interrogate-detrend-workbench.md` — workbench destination doc (gitignored)
- `../PLAN-analysis-lane.md` — thread C / task L5 (dev-viz), task H6 (speed pass)
- Vault hub: `/home/bgo/notes/bgovault/2.Areas/VI_GPS_Library/1776347706-gps-library-ecosystem-hub.md`

---

*Last reviewed: 2026-10-31 — history in `git log --follow CLAUDE.md docs/`.*
