# plots.yaml — the standard-plot products (design, 2026-10-21)

Replaces the 24 `plot-<area>.sh` cron scripts on gpsplot with ONE config file
and ONE runner. Laptop first; viewed through a local aflogun before anything
runs operationally.

## What gpsplot does today (read 2026-10-21, `gpsops@gpsplot:~/bin`)

| | today |
|---|---|
| local figdirs | `~/gamit-times/figures/` (standard) and `figures/<system>/` (per area) |
| publish, CDN | `/data/gps/timeseries/graphs/standard_24h/` and `/data/gps/locations/volcanos/<system>/graphs/24h/` |
| publish, brunnur | `/srv/www/brunnur/gps/{timeseries,eldfjoll/<system>,skridurogflod/<system>}/` |
| products | plate 90d/year (+full for some) per area; standard 90d/year + `--ref all` full |
| per-area knobs | `-u` 10–150, `--ylim` 15–50, fixedstart dates, Reykjanes event list |
| station lists | hardcoded per script (duplicates station_areas.yaml) |
| publish | `rsync ${figdir}/*.png` after every run (indiscriminate globs) |
| scheduling | 22 hand-staggered cron lines; one log file per area, some misnamed |

aflogun already shows these PNGs: it lists the CDN directories (nginx
autoindex) at `[PATHS] aflogun_graphs_standard_url` and
`aflogun_graphs_volcanos_url/<cdn-slug>/graphs/24h`, and labels each file by
its name after `STA-` — a new product (e.g. `-detrend-cleaned-90d`) appears
in its popup with no code change.

## Proposal

### 1. One plot tree that mirrors the CDN

```
<plots_dir>/                                   [PATHS] plots_dir (laptop: ~/gps-data/plots)
  timeseries/graphs/standard_24h/STA-*.png     = cdn /data/gps/timeseries/graphs/standard_24h
  locations/volcanos/<cdn>/graphs/24h/STA-*.png = cdn /data/gps/locations/volcanos/<cdn>/graphs/24h
  _runs/<UTC-stamp>/<product>.manifest         every file the run wrote
  _runs/<UTC-stamp>/publish.manifest           only publish: true runs (rsync input)
  _runs/<UTC-stamp>/run.log                    one log per run, all products
```

- The tree IS the publish layout, so publishing is one
  `rsync --files-from=publish.manifest` per target, never a glob.
- Areas are named by their `cdn:` slug in station_areas.yaml — the slug
  aflogun already uses — so there is one name per area everywhere.
- 8 h and detrended siblings live in the same directory; the file name says
  what they are (`STA-plate-8h-90d.png`, `STA-detrend-cleaned-year.png`).
- `_runs/` keeps the last N runs (pruned by the runner), not 19 GB of logs.

### 2. plots.yaml (gps-config-data → `~/.config/gpsconfig/plots.yaml`)

```yaml
plots_dir: null                 # null = [PATHS] plots_dir
keep_runs: 14
defaults: {frame: itrf14, save: png, legacy_names: true}

products:
  standard:
    stations: active
    dir: timeseries/graphs/standard_24h
    runs:
      - {ref: plate,   special: [90d, year, full], ylim: 50, publish: true}
      - {ref: detrend, special: [90d, year, full], operational: true}  # publish: false

  areas:                        # one product per station_areas.yaml area with cdn:
    stations: area              # members from station_areas.yaml
    dir: locations/volcanos/{cdn}/graphs/24h
    runs:
      - {ref: plate,   special: [90d, year], ylim: 20, publish: true}
      - {ref: detrend, special: [90d, year], operational: true}
      - {session: 8h,  frame: itrf20, ref: plate, special: [90d]}   # 8 h exists only in itrf20
    events: catalog             # events.yaml rows whose region == the area
    overrides:                  # only what differs per area
      reykjanes: {ylim: 15, fixedstart: [2023-11-12, 2020-01-01]}
      askja:     {fixedstart: [2021-07-19]}
      eskaftarketill: {ylim: 5000}
```

`publish` (default **false**) gates the CDN: the runner writes every run's
files, but only `publish: true` runs go into the publish manifest. New
products (detrended siblings, 8 h) are viewed locally until BGÓ flips them.

The legacy `-u` (uncertainty cut, 10…1 000 000) and `--ylim` values were tuned
for the py2.7 TOT reader; they are NOT ported verbatim. `uncert` is re-derived
on the .pos lane per area (the cleaned view does most of what `-u` did), and
Reykjanes is checked side by side with today's CDN plot first.

Station lists live ONLY in station_areas.yaml; event markers ONLY in
events.yaml (region + kind). The catalog has ~34 Reykjanes rows in 6 kinds
where the legacy list has 19 dates; drawing catalog rows as lines is new code
(`addEvent` takes only the CLI dict today).

### 3. Runner: `gpsplot run [plots.yaml] [--product P] [--area A] [--dry-run]`

In-process (no subprocess per area): expands products → (stations, variants),
runs them through the same batch contract as `gpsplot ts` (strict, one-line
errors), writes one manifest per product, prunes `_runs/`, exit 1 if anything
failed. `--dry-run` prints the expanded plan (what would be plotted where).

### 4. Viewing on the laptop

- `python -m http.server -d ~/gps-data/plots 8070` — checked 2026-10-21: its
  listings parse with aflogun's own `parse_files`/`parse_dirs` regexes.
- aflogun on a local branch `feat/local-plots` (from aflogun `main`) with
  `aflogun_graphs_standard_url = http://localhost:8070/timeseries/graphs/standard_24h`
  and `aflogun_graphs_volcanos_url = http://localhost:8070/locations/volcanos`.
  aflogun reads these only from `[PATHS]` in postprocess.cfg (no env
  override), so: a layered `GPS_CONFIG_PATH` test config (as
  `tools/local-plot/setup-testcfg.sh` does) — never a hand edit of the
  deployed postprocess.cfg. The branch is for what the new products need in
  the UI (grouping 90d/year/full per ref, the 8 h and detrended siblings).

### 5. Then operational

Same plots.yaml on the server; a systemd timer runs `gpsplot run` after
GLOBK; a publish step rsyncs the manifests to the CDN. Legacy cron lines are
commented out area by area once BGÓ signs off the side-by-side.

## Not plots, same crontab (handle before gpsplot is drained)

`save-gps-timeseries.sh` (8–12 h), `files4LavaCentre.sh`,
`plot-hekla-multigas.sh` (gas, hourly), `gpsplot-config-sync.sh`, `plot-eldv.sh`
(to `brunnur:~/tryggvi/eldvatn`, not in cron).

## Decisions (BGÓ 2026-10-21)

1. brunnur: NOT published to — aflogun replaces the brunnur pages; CDN only.
2. 8 h products go in the same `graphs/24h/` directory (named `-8h-`).
3. `volcanos/thorbjorn` and `bardarbunga-special` are retired.
4. Event lines from events.yaml: all kinds — eruption (red), dike/intrusion
   (blue), inflation/deflation, earthquake/swarm (colours set in plots.yaml).
5. The standard `full` plot with no plate removed (`--ref all`'s itrf) is dropped.

## Run order (2026-10-22)

`prepare:` lists steps `gpsplot run` performs first, in order, before any
product (`PREPARE_STEPS` in `cli/run.py`). `network_days` runs
`gps-network-days`: the network-day and network common-mode lists in
`pos_dir`, which the cleaned view removes and subtracts — so they must be
current before a plot is drawn. A failed step ends the run with no plots;
`--no-prepare` skips the steps (a quick re-run of one area); `--dry-run`
lists them.

## Free ts options: `args:` (2026-10-22)

Any `gpsplot ts` option can be given as `args:` — a string or a list of
strings, each shell-split (`"--outlier-sigma 3.5"` is two tokens) — at four
levels, appended general to specific after the generated options:
`defaults` → the product → the run → an area override. argparse keeps the
last value of a single-value option, so the most specific level wins.

```yaml
defaults: {frame: itrf14, args: [--traceback]}
products:
  areas:
    args: [--show-icing, --show-snow]               # every run of the product
    runs:
      - {ref: detrend, special: [90d], operational: true,
         args: ["--outlier-sigma 3", --show-outliers]}
    overrides:
      reykjanes: {args: ["--outlier-sigma 2.5"]}   # this area, every run
```

The typed keys (`ref`, `ylim`, …) stay; `args:` is for everything else.
Refused (set elsewhere): `-d/--figDir` (the product's `dir:`), `--manifest`,
`-j/--workers` (`gpsplot run -j`), `--special` (`special:`), `-s/--start`
(`fixedstart:`). `gpsplot run` parses every job's options before anything
runs (prepare included): an unknown option or a bad value ends the run at
the start, exit 2. `--dry-run` shows the full ts line of each job.

`operational: true` fills its presets only into options left at their
default, so an `args:` value wins — except one equal to the ts default
(`--outlier-sigma 4.5` with `operational` still gives 3.5); drop
`operational` and spell the options out for that.
