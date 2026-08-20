# gps_plot

Time-series visualization for GPS displacement data and auxiliary geophysical signals.
Legacy maintenance — used for research plots and dashboard exports — plus the new
**analysis-lane dev-viz** (`dev_viz.py`, PLAN-analysis-lane §4 L5).

## Status

Legacy modules: maintenance only, ~3362 LOC; two dated snapshots are kept as
historical reference (do not delete without coordinating). Modern lane
(2026-07→): `dev_viz.py`, `maps.py`, `detrend_workbench.py` + the two
pickers — each routed to its own doc below.

**Quality gates.** `uv run ruff check` / `ruff format --check` / `mypy src`
are enforced by `.github/workflows/ci.yml`; `uv run pytest` is local-only
(the suite reads real station data and deployed catalogs, so it cannot run on
a stock runner). mypy is `strict`, scoped to the same modern lane as the ruff
excludes — legacy plotting modules are `ignore_errors`, `tests/` is out. Scope
rationale, the 137 → 0 path, and an **open cross-package blocker** (CI is red
until `gps_analysis`'s branch reaches `main`): `docs/mypy-status.md`.

## Layout

```
gps_plot/
├── src/gps_plot/
│   ├── plot_gps_timeseries.py        # main plot driver (console script)
│   ├── timesmatplt.py                # time-series matplotlib plots
│   ├── gasmatplt.py                  # GAS plots (legacy, ruff-excluded)
│   ├── gmtplot.py                    # dormant velomap seed — absorbed into maps.py
│   ├── maps.py                       # PyGMT map lane (optional 'maps' extra)
│   ├── dev_viz.py                    # analysis-lane dev-viz (gps_analysis outputs)
│   ├── detrend_workbench.py          # operator workbench — the only config-WRITING module
│   ├── detrend_picker_qt.py          # Qt picker — emits a workbench command
│   ├── detrend_picker.py             # marimo picker — same contract, older
│   └── gasmatplt_{bgo,workingon}15May17.py  # Py-2 snapshots — DO NOT delete
├── docs/                             # detrend-lane, map-lane, dev-viz, mypy-status
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
gps_plot                  # entry: gps_plot:main
plot-gps-timeseries ...   # entry: gps_plot.plot_gps_timeseries:main
gps-analysis-devviz ...   # entry: gps_plot.dev_viz:main (dev group required)
gps-detrend-workbench ... # entry: gps_plot.detrend_workbench:main
gps-detrend-picker-qt ... # entry: gps_plot.detrend_picker_qt:main (pyqtgraph)
gps-detrend-picker ...    # entry: gps_plot.detrend_picker:main (marimo notebook)
```

## Detrend lane → `docs/detrend-lane.md` 📄

The cleaned view, the detrend workbench, segments and both pickers live in
**[`docs/detrend-lane.md`](docs/detrend-lane.md)** — measured behaviour that
belongs one level down. Read it before touching any of them.

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
  figure.** Eight violations of it are catalogued there, all the same shape —
  a second place assembling the same decision. Both pickers now build their
  run-flag tail with `detrend_workbench.run_flags`, because two pickers
  assembling it independently forgot the same flag independently.

**Per-station curation is config, not just a record.** `--commit` writes to
three places a batch re-run must find — `detrend_params.json` plus two
`analysis.yaml` keys — because `gps-estimate-detrend` RECOMPUTES the record
and cannot see a decision that lives only inside it. Detail in the lane doc.

## Dev-viz → `docs/dev-viz.md` 📄

`dev_viz.py` — three shared-axis panels for one station/component (observed +
`lineperiodic` fit with WLS velocity annotation, detrended residuals, GBIS4TS
break model). All math is called from `gps_analysis`; nothing is re-derived
here. Two things to know before using it, both in
**[`docs/dev-viz.md`](docs/dev-viz.md)**: `--break-input` decides what
detection runs on and one of its three values is known-bad on purpose, and the
MCMC dev default is 2 000 iterations against production's 1e6 — the dev
posterior is indicative only.

```bash
uv run gps-analysis-devviz --synthetic --component east --out /tmp/devviz.png
uv run gps-analysis-devviz --neu STA.NEU --component north --breaks 2 --out sta.png
```

## Map lane → `docs/map-lane.md` 📄

`maps.py` — four PyGMT functions sharing one set of slice-1 conventions
(coords from `stations.cfg` via `gps_parser`, zero hardcoding, lazy pygmt
guard, `dem_grid=` hillshade on all of them): `station_map`, `velocity_map`,
`deformation_vectors`, `slip_map`. They consume `gps_api` products directly,
and the dormant `gmtplot.py::velomap` recipe is fully absorbed into them.
Signatures, inputs and the install story are in
**[`docs/map-lane.md`](docs/map-lane.md)**.

```python
from gps_plot.maps import station_map
station_map(["RHOF", "AKUR"], title="North Iceland", outfile="stations.png")
```

## Cross-References

- `../CLAUDE.md` — ecosystem overview + dependency graph
- `../geo_dataread/CLAUDE.md` — produces the time series this package plots
- `../gps_analysis/CLAUDE.md` — the leaf math library dev_viz visualizes
- **[`docs/detrend-lane.md`](docs/detrend-lane.md)** 📄 — cleaned view, workbench,
  segments, Qt picker (the deep lane; read before touching any of them)
- **[`docs/map-lane.md`](docs/map-lane.md)** 📄 — the four PyGMT map functions,
  their `gps_api` inputs, and the GMT install story
- **[`docs/dev-viz.md`](docs/dev-viz.md)** 📄 — `--break-input`, MCMC dev defaults
- **[`docs/mypy-status.md`](docs/mypy-status.md)** 📄 — the typing gate's scope,
  the 137 → 0 path, and the open `gps_analysis` blocker
- `tools/local-plot/README.md` — local figure workflow, TOT join, epoch/shell traps
- `.interrogate-detrend-workbench.md` — workbench destination doc (gitignored)
- `../PLAN-analysis-lane.md` — thread C / task L5 (dev-viz), task H6 (speed pass)
- Vault hub: `/home/bgo/notes/bgovault/2.Areas/VI_GPS_Library/1776347706-gps-library-ecosystem-hub.md`

---

*Last reviewed: 2026-08-18 — map lane routed out to `docs/map-lane.md` and this
footer's accumulated history dropped, which is what brought the file back under
the 150-line budget. That history was a changelog restating commit messages;
`git log --follow CLAUDE.md docs/` is the authoritative version and does not go
stale. The measured behaviour those entries pointed at was never in here — it
lives in `docs/detrend-lane.md` and `docs/map-lane.md`, which is where it stays.*
