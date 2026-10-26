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
│   ├── rtk.py                        # gpsplot rtk: RTK baselines (geo_dataread.rtk_read), --daily GLOBK join
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
gpsplot [ts] STA ...      # entry: gps_plot.cli:main (ts is the default subcommand)
gps_plot ...              # same as gpsplot
plot-gps-timeseries ...   # old name of gpsplot ts, kept until cron/scripts move
gps-analysis-devviz ...   # entry: gps_plot.dev_viz:main (dev group required)
gps-detrend-workbench ... # entry: gps_plot.detrend_workbench:main
gps-detrend-picker-qt ... # entry: gps_plot.detrend_picker_qt:main (pyqtgraph)
```

## gpsplot subcommands → `docs/cli-subcommands-design.md` 📄

One command, `gpsplot`, one subcommand per plot kind (`cli/__init__.py`
`COMMANDS`; only the chosen one is imported). `ts` options live once in
`plot_gps_timeseries.build_parser`, shared with the old name. `--model
curve|residual|both` draws the stored model over the plate / detrended
series (line = what the model still has in the data, in the window's datum;
`--model-terms` picks the line's terms independently of the data's). `--rate` / `--rate-window
A:B` (repeatable) draw the velocity product's rate per window
(`layers/rate.py`: line with the product's slope through the plotted points,
`v = rate ± σ` per panel, or the refusal reason; files `-rate`): `--ref plate`
→ plate, `--ref detrend` → background (`--detrend-terms periodic` →
plate-periodic) — the map's and `gps-velocities`' numbers for the same ref
and view (they default to plate-periodic + cleaned; `ts` follows the plot). `--rate-segments steps|events` = one rate per span between events (design doc). `--view cleaned` detects outliers with the prior-free running window by default (gps_analysis design §15; `--outlier-sigma K`, default 4.5; σ not in the file name); `--outlier-method model` (or `--stages`) = the old trajectory-fit detector, file gets `-modeldetect`. Antenna obstruction, two indices: icing days (GAMIT autcln index, `gps-autcln-harvest`; all components, light blue) and snow-burial days (geo_dataread.snow_burial; Up only, pale lilac) — each `--icing`/`--snow remove|keep|off` (default remove) and `--show-icing`/`--show-snow [N]` (independent of `--show-outliers`); `--obstruction` / `--show-obstruction [N]` set both (an explicit per-kind flag wins); `--icing-ratio R`. Network days (whole network solution bad that day; `gps-network-days` list in pos_dir): `--network remove|keep|off` (default remove, every component), `--show-network [N]` draws them dark grey (`--operational` shows all). `--common-mode subtract|off` (default subtract): the daily network common mode (`network_cm.itrf14.csv`) is subtracted before detection on okada 24 h itrf08/itrf14 series only; high-pass, so rates/seasonals/records are unchanged; note "network common mode removed".
**Production batch contract (2026-10-21):** `gpsplot ts active` = the production set (stations.cfg, not passive/discontinued/inactive, in the box, with a series); one line per failure + a summary, exit 1 if any plot failed (no data in the period = skip, not failure); `--strict` refuses degraded output (raw under -cleaned/-detrend, no plate) before writing; `--manifest FILE` lists the files written (publish these, never glob); `--legacy-names` drops the frame tag (`STA-plate-90d.png`); `--save` alone = png. `--quake TIME,LAT,LON,MAG[,NAME]` marks an earthquake (layers/quakes.py). **`gpsplot run [STA ...]`** expands `plots.yaml` (`products.py`; stations narrow the selected jobs; baselines from station_areas.yaml `baselines:` (pairs), stacks from its `stacks:` (groups), plot roots from its `category_roots:`; design + `args:`/`prepare:` in `docs/plots-yaml-design.md`).
`gpsplot ts all` = every station with a .pos series of the frame in the Iceland box (geo_dataread `pos_stations`; TOT: stations.cfg); `--region=W/E/S/N` (with `=`, west is negative) replaces the box and filters lists, `--with-record` keeps stations with a detrend record.
`gpsplot stack "SEY?-SEYD" HOFN --offset 20` overlays stations and baselines
(patterns match stations.cfg). `--events YYYYMMDD[,COLOR] …` / `--eventf FILE` draw event lines on every
panel in `ts`, `stack` and `rtk` alike (`cli/common.add_event_options`; not
`seasonal`, which folds onto one year). A late-starting stack series is backcast
along a robust line (MAD-clipped, `stack._rate_line`) and a detrended series is
zeroed on its first clean samples (`sources.gps.first_clean`): one undetected
outlier at the start no longer sets the zero (OLAC, July 2021). `gpsplot seasonal NYLA` folds a station's seasonal onto one year: data, own, regional LOO (design doc). `gpsplot rtk "SEY?-SEYD" [--daily]` plots the
RTK baselines of rtk.vedur.is (+ the daily GLOBK ones). `gpsplot map velocity
--region W/E/S/N --window A:B --window C:D` draws windowed velocity fits per
window (BGÓ's Reykjanes maps); also `station`, `deformation`, `slip`.
Planned: seismic panels (on ice) — the staged plan and BGÓ's
decisions are in the design doc. Every step is gated by
`bin/check_cli_variants.py` (byte-identical PNGs vs a `main` export).

## Detrend lane → `docs/detrend-lane.md` 📄

The cleaned view, the detrend workbench, segments and both pickers live in
**[`docs/detrend-lane.md`](docs/detrend-lane.md)** — measured behaviour that
belongs one level down. Read it before touching any of them.

- **Cleaned view** (`--view cleaned`) — red clean, grey flagged (NaN,
  removed), **gold provisional** (kept, verdict pending). Loosening
  `window_n_sigma` is NOT monotonic: zero flagged = the excess-candidate
  rule ABORTED and you are being served raw data.
- **Detrend workbench** (`gps-detrend-workbench`) — detrend choice is
  curation, not computation. The only config-WRITING module in the package:
  it merge-writes the record (`detrend_params.json`), the declarations
  (`steps.yaml`, via `--declare-step`) and the `analysis.yaml` keys.
  `--frame itrf14` (default since 2026-10-22: `.pos` itrf14 +
  `detrend_params.itrf14.json`); `--frame tot` = the old joined-TOT path.
- **Segments** (`--segment A:B`, repeatable) — the fit domain is a union of
  intervals (answers four asks at once); the gates changed meaning per segment.
- **Qt picker** (`gps-detrend-picker-qt`) — the one GUI, layered ON TOP of
  the CLI; the invariant is **the emitted command reproduces the figure**,
  and a `store…` button runs that command in-process — one write path, the
  workbench's own. Eight catalogued violations, all the same shape: a
  second place assembling the same decision (lane doc).
- **The compositional model** (`f(t) = Σ mᵢ(t)`, 2026-08-22) — N stage cards,
  membership split from assignment, the peel that follows the active stage,
  `--final joint` and the commit mode the batch forces. The grammar already
  expressed all of it; this was UI exposure and storage.
- **Cross-station borrowing** (2026-08-26) — a `store:STA` borrow is
  RE-ANCHORED to this station's level (`--anchor-window`, default full fit
  span; window shown in summary + record provenance); `--stage apply:` +
  holds = the fully-borrowed station (ELDC). Numbers + batch gap: lane doc.
- **Qt picker, background phase: `or borrow s(t) from:`** (2026-08-26) — the
  control for a station that has NO clean interval to fit on, so the phase's
  own question has no answer for it. Emits `borrow_command` (`--stage apply:`
  + a `store:` hold per group); the terms tickboxes choose which groups are
  borrowed, and unticking one frees it to be fitted here instead. ONE donor
  field for all three components on purpose — a secular velocity is one
  3-vector. Eagerly resolved, so a missing donor or a **plate-frame
  mismatch** is refused in the panel, not at fit time. **Defaults to
  seasonal-only** (the legacy detrend-OLAC manoeuvre: no datum and no rate
  cross), applied once on the transition into borrowing; holding `secular`
  too is the deliberate choice, for a station whose own trend IS deformation.

**Per-station curation is config, not just a record.** `--commit` writes to
three places a batch re-run must find — `detrend_params.json` plus two
`analysis.yaml` keys — because `gps-estimate-detrend` RECOMPUTES the record
and cannot see a decision that lives only inside it. Detail in the lane doc.

## Baselines (2026-10-08)

`plot-gps-timeseries VFLN-VFLS` plots the **baseline** A − B instead of one
station: N/E/U differences on common days (matched within half a day, never
by float equality), re-zeroed at the first common epoch, σ combined in
quadrature (conservative — common-mode errors are counted twice).
**Layering, for reuse:** everything that is not drawing lives in
`geo_dataread.baseline` (numpy only — parsing, day matching, the difference,
both ends referenced alike, the mixed-view check), so `gps_api` or a notebook
forms the same baseline from `read_gps_view` output. gps_plot keeps only
presentation: titles, `baseline_file_stem`, `add_baseline_ends(fig, labels,
ends)` (works on any 3-axis figure). Both ends of a plotted baseline go
through ONE per-station pipeline, `timesmatplt.station_series` (which
partly duplicates `read_gps_view` — unifying them is a separate refactor).
Defaults to `--ref plate`; every `--ref`/`--view` works. A **detrended**
baseline refuses an end without a stored record (it would difference
detrended against plate). Titles name both stations and both plates when
they differ; files are `VFLN_VFLS-baseline-<ref>…` (underscore, because the
publish convention splits on `-`). `--show-ends` also draws both stations'
own series behind the baseline (A light grey, B dark grey, both zeroed at the
baseline's first common epoch, so red = their visible gap; files `-ends`).
The same change made `--view detrended`
work at all — plotTime used to reject it — and tags its files `-detrended`.

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
`gpsplot map station --tos` plots every TOS GPS station by status
(`--tos-search "…"` takes any `tos search` line, `--stations-file` adds
sites from files, `map base`/`--no-stations` draws none), and
`--dem iceland` draws the spliced Iceland DEM at the level the map needs
(the TOS current-value rules are in the lane doc). Regional maps
(2026-10-16): `--region reykjanes`, `--layers reykjanes` (roads, towns,
lavas, graben, fissures, barriers, places), `--inset svartsengi` (zoom box),
the Iceland locator and decluttered station names. Windowed velocity maps
read the velocity product (`geo_dataread.velocity`; `--velocities FILE`,
`--velocities-out DIR`, `--ref plate-periodic|background|plate`). Signatures, inputs and
the install story are in **[`docs/map-lane.md`](docs/map-lane.md)**.

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
  the 137 → 0 path, and the (now cleared) `gps_analysis` blocker
- `tools/local-plot/README.md` — local figure workflow, TOT join, epoch/shell traps
- `.interrogate-detrend-workbench.md` — workbench destination doc (gitignored)
- `../PLAN-analysis-lane.md` — thread C / task L5 (dev-viz), task H6 (speed pass)
- Vault hub: `/home/bgo/notes/bgovault/2.Areas/VI_GPS_Library/1776347706-gps-library-ecosystem-hub.md`

---

*Last reviewed: 2026-10-23 — plots.yaml baselines + stacks (area_baselines/area_stacks, stack --label/--manifest); 2026-10-22 — gpsplot run STA ... (station subset); workbench/picker --frame (itrf14 default); gpsplot run: plots.yaml `args:` (free ts options, defaults→product→run→override; checked before the run), `prepare:`; 2026-10-21 — ts --network/--show-network (network days, dark grey); ts active/--strict/--manifest/--legacy-names/--quake, exit status; 2026-10-20 — ts all/--region/--with-record; snow-burial flags + --obstruction umbrella; 2026-10-19 — gpsplot seasonal, ts --rate-segments, --outlier-method running/--outlier-sigma, icing index; 2026-10-16 — velocity maps read the velocity product; regional maps: --layers, --inset, locator, decluttered names, fancy scale; 2026-10-15 — map --tos, --tos-search, --stations-file, --dem iceland; 2026-10-14 — gpsplot subcommand CLI, --model, stack, rtk, map; 2026-10-08 — baselines (AAAA-BBBB) + the --view detrended fix;
earlier 2026-08-26: cross-station borrow re-anchoring +
`--anchor-window` + apply-only plans (detail routed to
`docs/detrend-lane.md`, per the 2026-08-18 rule: measured behaviour lives in
the lane docs, history in `git log --follow CLAUDE.md docs/`).*
