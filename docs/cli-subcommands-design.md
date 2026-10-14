# gps_plot: subcommand CLI and modular plotting — design

*Status: proposal, 2026-10-13. Review by Fable 5.1 of BGÓ's request (two passes; the
second after BGÓ freed the command name), edited after the `feat/ref-frame-option`
merge (main `ff4e1f4`). Steps 0–3 done 2026-10-14.*

## Decisions (BGÓ, 2026-10-14)

1. Command `gpsplot`, subcommand `ts` (default); `plot-gps-timeseries` stays as
   an alias until cron/scripts are moved.
2. SEY?-SEYD: **both** RTK and daily GLOBK baselines, independent sources — each
   can be plotted on its own (as rtk_gps does / as the daily baselines) and they
   can be joined in one plot.
3. Stack: overlay with legend by default, `--offset MM`.
4. `--model curve|residual|both`; `both` = two files first; refuse `--ref ref_frame`.
5. Seismic: SIL files behind an adapter now, skjálftalísa later.

## Done

- **Step 0**: `bin/check_cli_variants.py` renders 8 variant sets (43 PNGs: frames,
  refs, cleaned + `--show-outliers 5`, `--detrend-terms periodic`, `--session both`,
  a baseline with ends, the production `--ref all --special all` fan-out) from a
  `main` export and compares sha256 against the candidate tree.
- **Step 1**: `gpsplot` (`cli/__init__.py`, `cli/ts.py`, `cli/tools.py`);
  `plot_gps_timeseries.main` split into `build_parser` + `run` (body unchanged).
  Gate: `gpsplot ts`, bare `gpsplot` and `plot-gps-timeseries` all byte-identical
  to `main` (43/43). The option code stayed in `plot_gps_timeseries.py` rather
  than moving to `cli/ts.py`/`cli/common.py`: shared option groups come with
  `stack` (step 4), when there is a second user for them.
- **Step 2** (2026-10-14, branch `feat/series-model`):
  - gps_analysis `select_record_groups` + `groups=` on `apply_detrend` /
    `evaluate_record` (staged groups: `secular` = offset + rate WITHOUT steps);
    geo_dataread `apply_stored_detrend(groups=)`; gps_plot's hand-made step
    add-back in `station_series` is gone (`DETREND_TERM_GROUPS`). Arrays agree
    with the old path to 3e-14 mm.
  - `series.Series` (frozen, arrays passed through untouched) and
    `sources/gps.py`: `SeriesRequest`, `BaselineOptions`, `fetch(spec, req)` for
    a station or a baseline (`ends`, empirical σ in `meta`). `plotTime` and the
    8 h overlay both read through `fetch`, so the station-or-baseline code
    exists once.
  - Gate: 17 variant sets, 54 PNGs byte-identical to main (stepped SELF/HOFN
    records, `--remove-steps`, empirical σ, `SENG-SKSH` with `--session both`
    and `--ref detrend`, `--outlier-param`, `--stages`); an empty variant now
    fails the gate. Every deployed record (3 documents × 7 `--detrend-terms`
    combinations) gives the same detrend through `groups=` to 3e-14 mm.
  - Found on the way: `--stages` without `--outlier-param` crashed on main
    (`'NoneType' object is not iterable`); fixed. `SeriesRequest.kwargs()` is
    shallow on purpose (`asdict` would turn OutlierParams into a dict).
- **Step 3** (2026-10-14, branch `feat/model-curve`): `gpsplot ts --model
  curve|residual|both`.
  - **What the line is** (refines decision 4): the part of the stored model
    still in the plotted data. `curve` = over the plate series, the full
    trajectory (all groups, transient included) minus any `--remove-steps`
    kinds; `residual` = over the `--ref detrend` series, what the
    `--detrend-terms` left in (the steps, unless removed), so it is flat
    (at the window's datum) when nothing is left. `data − line` is then the same
    residual in every view. Built by running `station_series`' own removal
    step (`timesmatplt.trajectory_views`, factored out) on zeros, so line
    and data cannot disagree.
  - **Window datum**: `getData` zeroes a series on its first kept samples
    and anchors the plate motion at its first epoch, inside the window; the
    record is fitted to the full span. So the line is shifted by
    geo_dataread `gps_views.window_datum_shift` (exact: full − window over
    common epochs, spread 0). Without it HOFN from 2024 is off by
    64/−143/−350 mm. Existing `--ref detrend` plots with `--special` carry
    that constant too (unchanged here, pixel gate); their residual is
    not zero-centred.
  - Drawing: `trajectory.py` (the workbench's dense, step-bracketed
    `trajectory_curve`, moved; `curve_grid` split out), `layers/model.py`:
    solid in the record's fit window, dashed where extrapolated; the data
    keep the axis scale, and a panel where the whole line is off scale says
    so with its level (SENG's pre-unrest background in the year view: N
    −1263, E +2591, U −1044 mm — the 2023 dike, not an error). Files
    `-model` / `-resid`; `both` = two files. Refused: `--ref ref_frame`/`all`
    or a mismatched `--ref`, baselines, and (follow-up) 8 h sessions /
    `--session both` / a reference position: those series are zeroed elsewhere
    and share no epoch with the daily read the record is fitted to, so no
    exact shift exists (seen: HOFN `--session both` line ~118 mm off).
  - Tests: the acceptance identity `data − line = D_full(plate) − M` at the
    data epochs on the deployed SELF itrf14 record (step 2008.41): full,
    2007–2010 and 2024– windows × raw/cleaned × plate/detrend × with/without
    `--remove-steps` (15 cases). Gate: the step-2 variants still identical,
    plus 4 candidate-only `--model` variants that must render their PNG
    count.
  - `--model-terms TERM[,TERM]` (BGÓ, same day; branch `feat/model-terms`):
    how the model is DRAWN is independent of how the data are shown. Given,
    the line is exactly those groups (linear, periodic, steps, transient →
    staged secular/periodic/step/transient) in the window's datum, whatever
    `--detrend-terms`/`--remove-steps` took out of the data — e.g. the
    seasonal line over data with the seasonal removed. Not given: the
    default above. Files add `-line-<terms>`.
- **Step 4** (2026-10-14, branch `feat/stack`): `gpsplot stack`.
  - `gpsplot stack "SEY?-SEYD" [STA …] --ref plate|detrend|ref_frame
    [--offset MM]`: stations and baselines in one N/E/U figure, each through
    `sources.gps.fetch` (the same read/clean/detrend as `ts`), overlaid with a
    legend; `--offset` moves series k by k·MM (named in the legend).
  - Patterns (`? * [..]`) match `stations.cfg`, on either side of a baseline
    (split at a `-` outside `[..]`, so `SEY[1-4]-SEYD` works); self-pairs
    dropped; a pattern matching nothing is an error. Quote them in zsh.
  - Colours are keyed on the label (crc32 into a 12-colour palette, next
    free slot on a clash within the figure): a station keeps its colour
    between plots unless it clashes with another in the same figure.
  - Failures: a series that fails is dropped and listed on stderr; nothing
    plotted = exit 1. A `--ref detrend` station without a stored record is
    plotted raw and labelled "(not detrended)" — never passes as detrended.
  - Detrended stations are centred: `sources.gps.centre_detrended` adds the
    exact window shift back, so the series is `data − model` about zero for
    any window (without it HOFN's year view sat ~65/−150/−375 mm off).
    `ts --ref detrend` still has that constant (pixel gate); fixing it
    there is a visible change, for BGÓ to decide.
  - Files `stack_<first>_<n>-<ref>[-terms][-cleaned][-offsetMM]-<frame>-<period>`
    (no `-` before the ref: publish globs split there).
  - Shared options moved to `cli/common.py` (period, output, series groups,
    `resolve_dates`, `resolve_source`); `ts` builds from them too, pinned by
    `tests/test_ts_kwargs_snapshot.py` (the exact plotTime kwargs for 18
    command lines + error messages). `timesmatplt.resolve_period` split out
    of `plotTime`. Composition: `compose.stack_figure` (stdFrame + layers).
  - Not yet: `--model` in a stack, rows per series, `--session both`,
    baselines re-zeroed on one (possibly noisy) first common epoch (as `ts`).

## Request (BGÓ)

Split `plot-gps-timeseries` into subcommands (as tostools/receivers) and make it
expandable: stack several series, stack baselines (SEY?-SEYD), draw the detrend
(trajectory) function as a line over raw and over detrended data, rtk_gps, PyGMT
maps, seismic activity, more later. Plotting only: analysis stays in
gps_analysis/geo_dataread, later via gps_api. Keep the current functionality; the
command **name** may change (shorter is better).

## 1. Current state

- Console scripts (`pyproject.toml:47-52`): `gps_plot` (a "Hello" stub),
  `plot-gps-timeseries`, `gps-analysis-devviz`, `gps-detrend-workbench`,
  `gps-detrend-picker-qt`. `maps.py` (PyGMT) has no CLI; `gmtplot.py` is dormant.
- `plot_gps_timeseries.py` is one ~1075-line argparse `main()`: parser,
  post-processing and the process-pool batch in one body; `vars(args)` goes
  straight into `plotTime(**kwargs)`.
- `timesmatplt.station_series()` (`:498-678`) is the fetch pipeline (getData →
  cleaned mask → remove_declared_steps → apply_stored_detrend), partly duplicating
  `geo_dataread.gps_views.read_gps_view`. `:647-666` re-adds step amplitudes by
  hand — trajectory math inside the plot package.
- `plotTime()` (`:681-1175`) hand-wires every overlay as an `if` block (baseline
  ends, outliers, provisional, 8 h session, abort badge, JOIN, events, steps, logo)
  and builds filenames. `stdFrame`/`addData` are hard-wired to 3 N/E/U axes.
- Blocks to extensibility: a new layer = a new `if` + a new kwarg through argparse;
  no series type (tuples of `(yearf, data[3,N], Ddata[3,N])`); no 4th panel; the
  two legacy modules are mypy-ignored, so new code must live in new modules.
- Reusable now: `detrend_workbench.trajectory_curve()` (`:1701-1776`, dense grid
  via `gps_analysis.evaluate_record`) — the "detrend line"; `geo_dataread.baseline`;
  `rtk_gps.plot_rtk_neu` (`rtk_gps/rtk_gps/rtk_gps.py:172-298`) already stacks RTK
  baselines.
- Tests: no image golden masters in `tests/`; `test_cli_smoke` pins the flag
  surface; `bin/check_pixel_equivalence.py` (sha256 of PNGs from two source trees)
  is the byte-identity gate.

## 2. Critique

Sound: argparse subparsers is house style (`tostools/src/tostools/tos.py:39`,
`receivers/src/receivers/cli/main.py` with `set_defaults(func=…)` and shared
`add_*_flag` helpers; no click/typer). The plotting-only rule matches the PLAN.

Push back:
- **rtk_gps**: different deployment (poetry, py3.12, pinned mpl, libnfs, paramiko,
  5-min upload loop). Don't fold the tool in; lift its reader (`open_datafile`,
  `rtk_gps.py:45`) into a pure `geo_dataread.rtk_read`, add an `rtk` source here.
- **"SEY?-SEYD"** is the RTK baseline family in the reference blob
  (`gps_data_analyses/pygmt/seyd-test.py:16-24`), though
  `seydisfjordur/svinaf-daniel-paper.py:405,480` also differences daily GLOBK
  SEY4/SEYD. Both become "N series of one kind in one figure" once a series type
  exists.
- **Seismic is a data-source problem.** Event lines exist (`addEvent`), a bars panel
  is small. Missing is a typed catalogue reader: `geo_dataread.sil_read` is legacy
  pandas with a hard-coded `/mnt_data/sildata`; skjálftalísa is the planned source.
- **PyGMT for time series: no.** Keep it for maps; give `maps.py` a CLI.
- **Production outputs must stay byte-identical** through every refactor step
  (brunnur hotlinks `{STA}-plate-{90d,year}.png`); the pixel gate runs each stage.
- **"Highly flexible"**: three extension seams (sources, layers, commands) and stop
  there — no plugin bus, no YAML plot DSL.

## 3. Architecture

Command `gpsplot` (see Q1) with subcommands: `ts` (today's timeseries driver,
baselines via `AAAA-BBBB`), `stack`, `rtk`, `map {station,velocity,deformation,slip}`,
`seismic` (later), plus lazy aliases `devviz`, `workbench`, `picker`
(`gpsplot --help` never imports Qt/pygmt).

- **Default subcommand**: if `argv[1]` is neither a subcommand nor an option, `ts`
  is inserted, so `gpsplot HOFN --save pdf` works.
- A spec is a station or `AAAA-BBBB` (the `geo_dataread.baseline.split_baseline`
  grammar), so no separate `baseline` subcommand; `stack` takes mixed specs.
- `gps-detrend-workbench` / `gps-detrend-picker-qt` keep their names: the picker's
  invariant is that the *emitted command* reproduces the figure; renaming changes
  emitted commands.
- **Import rule** (to go in CLAUDE.md): `sources/` may import geo_dataread/gps_parser;
  `layers/` and the renderer may import neither (they get arrays and records); any
  math on a record goes to gps_analysis/geo_dataread. Registries are in-package dicts;
  an entry-points group only when an outside package actually needs to plug in.
- `plotTime` calls `mpl.use("WebAgg")` as a side effect (`:786`); move that to the
  CLI's interactive path.
- Keep `plotTime`, `station_series`, `stdTimesPlot`, `addData`, `saveFig`,
  `add_baseline_ends` importable with the same signatures (notebooks, workbench).

```
src/gps_plot/
  cli/__init__.py     # main: subparsers, lazy command import, set_defaults(func=)
  cli/common.py       # option groups defined ONCE: stations, window (-s/-e/-D/--special),
                      # product (--frame/--session/--pos-id), view (--ref/--view/
                      # --detrend-terms/--remove-steps/--show-outliers), output (--save/-d/--ylim)
  cli/ts.py           # today's main() moved verbatim
  cli/stack.py, cli/rtk.py, cli/map.py, cli/seismic.py
  series.py           # Series / EventCatalog dataclasses
  sources/{gps,baseline,rtk,seismic,api}.py   # thin adapters -> Series; NO math
  layers/{points,curve,events,bars,badge,logo}.py
  figure.py           # Panel / FigureSpec + matplotlib renderer (wraps stdFrame/addData)
  naming.py           # legacy filename policy, extracted from plotTime
```

```python
@dataclass(frozen=True)
class Series:
    label: str                      # "HOFN", "VFLN-VFLS", "SEY1-SEYD"
    t: FloatArray                   # yearf (N,)
    y: FloatArray                   # (C, N) mm
    sigma: FloatArray | None
    components: tuple[str, ...]     # ("north", "east", "up")
    frame: str | None
    ref: str
    masks: Mapping[str, BoolArray]  # "outlier", "provisional"
    meta: Mapping[str, Any]         # record, plate, pos_id, …

class SeriesSource(Protocol):
    def fetch(self, spec: str, req: SeriesRequest) -> Series: ...

class Layer(Protocol):
    def draw(self, axes: Sequence[Axes], ctx: RenderContext) -> None: ...

SOURCES / LAYERS / COMMANDS: registries, also loaded from entry points
("gps_plot.sources", "gps_plot.layers", "gps_plot.commands")
```

Adding a data type = one `sources/x.py` + a registry line; a plot kind = a `Layer`
or a `cli/x.py` composing layers. Sources call geo_dataread (later gps_api);
analysis stays in gps_analysis (`evaluate_record`, `apply_detrend`, `select_terms`).

## 4. Features mapped

| Feature | Design | Exists | Missing |
|---|---|---|---|
| Stack series | `gpsplot stack STA…`: overlay + legend, `--offset MM`; per-row panels later | `station_series` / `read_gps_view` | stack semantics (Q3) |
| Stack baselines | N `baseline` sources (GLOBK) or `rtk` sources | `geo_dataread.baseline`; `plot_rtk_neu` as reference | `geo_dataread.rtk_read` |
| Detrend line | `ModelCurveLayer` from `trajectory_curve` (moved to `layers/`, workbench imports it back). `ts --model curve` = stored trajectory over the plate series (the record is plate-frame; refuse with `--ref ref_frame` at first); `--model residual` = detrended with a zero line; `--model both` = two files (`-model`, `-resid`) first, side-by-side later | `gps_analysis.evaluate_record`, `select_terms`; examples `gps_data_analyses/karahnjukar/detrend_test.py:148-199`, `detrend-bb/bb-paper.py:560-620` | step add-back must move to gps_analysis/geo_dataread; `-model` filename suffix |
| rtk_gps | `sources/rtk.py` + `gpsplot rtk BASE-ROVER…` | `rtk_gps.open_datafile` | pure reader; cm→mm |
| PyGMT | `gpsplot map …` over `maps.py` | `maps.py` | CLI only |
| Seismic | `EventsLayer` + `BarsLayer` panel (count/day, cumulative moment on twin axis) via `--seismicity` | `sil_read.read_sil_data` (legacy); `tmp_multiploting/beta_multiplot.py`, `bb-paper.py:184,202` | typed catalogue reader, config paths; skjálftalísa later |
| API source | `sources/api.py` | gps_api series endpoint | trajectory endpoint |

## 5. Compatibility

All flags and behaviour of today's command move unchanged into `gpsplot ts`
(the body is moved, not rewritten): defaults, `--special`, `-d` from
`postprocess.cfg fig_dir`, process pool, filenames. `plot-gps-timeseries` can stay
as a thin alias (cron/scripts) or be dropped — BGÓ's call (Q1). The other console
scripts stay and also appear as subcommands. Tests: flag pin on the new command,
`--help` option-set equality if the alias stays, pixel gate on DYNG/ELDC/SENG
production variants every stage.

## 6. Roadmap

0. **Baseline capture, no code**: pixel hashes + produced file names for a fixed
   station set × variants (raw, cleaned+outliers, detrend, baseline, `--session both`,
   `--special 90d`), and a table test of today's file names before anything moves.
1. **CLI split, no behaviour change** (`cli/__init__`, `cli/common`, `cli/ts`). Low risk.
2. **Series model + gps source**; move the step add-back out of gps_plot
   (cross-repo). Medium risk. Array equality on goldenmaster fixtures + pixel gate.
3. **Model-curve layer**, opt-in `--model-curve`, `-model` suffix. Low risk.
4. **Panel composition + `stack`** (plotTime not migrated yet). Low-medium.
5. **Seismic**: typed catalogue reader in geo_dataread, `sources/seismic`, `layers/bars`. Medium.
6. **RTK**: `geo_dataread.rtk_read`, `gpsplot rtk`. Medium.
7. **Map CLI** over `maps.py`. Low.
8. **Migrate `plotTime` onto the composition engine** — gated by the pixel check
   across all production variants. High risk, done last.
9. **API source** once gps_api exposes the trajectory record.

## 7. Open questions (recommendation in bold)

1. Name: **`gpsplot`** (short, no shift-key) or `gps-plot`; keep `plot-gps-timeseries` as an alias? **Keep a thin alias until cron/scripts are moved, then drop.** Subcommand `ts` or `timeseries`? **`ts`, with `timeseries` as alias.**
2. SEY?-SEYD = RTK baselines, daily GLOBK, or both? **Both are covered; RTK first?**
3. Stack: overlay with legend, vertical offsets, or one row per station? **Overlay default, `--offset MM`, rows later.**
4. Model curve: `--model curve|residual|both`; `both` as two files first? **Yes.** Over `--ref ref_frame` (plate velocity added back) or refuse? **Refuse at first.**
5. Seismic source/product: SIL files now vs skjálftalísa; count/day bars, cumulative moment, magnitude stems within a radius of the station? **SIL files behind an adapter now; magnitude stems + cumulative moment on a twin axis (as bb-paper.py); the reader comes first.** Note: earthquake *markers* already exist (`--annotate-steps earthquake`).
6. rtk_gps: **stays separate; only its reader moves into geo_dataread.**
7. PyGMT for time series? **No, maps only.**
8. Pixel identity required for production variants every stage? **Yes.**
9. Move the step add-back math to gps_analysis/geo_dataread in step 2? **Yes.**
10. `Series` in gps_plot or geo_dataread? **gps_plot for now; promote if aflogun/gps_api want it.**
11. gps_api trajectory endpoint? **Yes, additive; not blocking.**
