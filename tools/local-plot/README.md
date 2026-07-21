# local-plot — laptop 3-view time-series workflow

Prove and iterate on the static-plot lane locally, publishing nothing:

```
gps-globk-tot          join okada GLOBK segments  ->  ~/gps-data/TOT
gps-estimate-detrend   fit stored-detrend params  ->  <testcfg>/detrend_params.json
plot-gps-timeseries    render raw | cleaned | detrended  ->  ~/gps-data/figs
```

The wrapper reads a **layered test config** (`setup-testcfg.sh`) so the deployed
`~/.config/gpsconfig` is never touched: base config files are symlinked (no
credentials), the analysis-lane catalogs are copied from the `gps-config-data`
checkout, and `detrend_params.json` is generated into the test dir.

**All generated output lives outside this repo** (`~/gps-data/…`); only these
scripts are versioned.

## Quick start

```bash
cd tools/local-plot
./setup-testcfg.sh                       # build ~/gps-data/testcfg (once, or after catalog edits)
ESTIMATE=1 ./plot-views.sh SKHA BLON      # fit detrend, then render all 3 views
./plot-views.sh AKUR                      # plot only (reuse existing detrend_params.json)
SAVE=pdf,png VIEWS=detrended ./plot-views.sh SKHA
```

Output filenames are siblings: `{STA}-plate.png`, `{STA}-plate-cleaned.png`,
`{STA}-detrend.png` (+ any `--save` vector formats — `pdf`/`eps` work natively).

## Prerequisites

- **geo_dataread on the Stage-B branch** (`feat/local-tot-pipeline`) — provides
  `gps-globk-tot` + `gps-estimate-detrend`. Once merged to `main`, the default
  `GDR=../geo_dataread` just works; until then point `GDR` at a branch checkout.
- **gps-config-data analysis-lane catalogs** including `segment_exclusions.csv`
  and `fit_windows.csv` (also Stage B). `setup-testcfg.sh` copies them from
  `GCD` (default `~/git/gps-config-data`); point `GCD` at the branch checkout
  until merged.
- `~/gps-data/TOT` populated by `gps-globk-tot` (192 stations as of 2026-07-21).

## Config knobs

Every path is env-overridable — see the header of each script. Common:

| var | default | meaning |
|-----|---------|---------|
| `TOT_DIR` | `~/gps-data/TOT` | local join output |
| `FIGDIR` | `~/gps-data/figs` | figure output (outside repo) |
| `TESTCFG` | `~/gps-data/testcfg` | layered test config |
| `SAVE` | `png` | `--save` formats (`pdf`, `eps,pdf,png`, …) |
| `VIEWS` | `raw cleaned detrended` | which views to render |
| `GDR` / `GCD` | siblings / `~/git/…` | package + config-data checkouts |

## fit_windows.csv — per-station detrend windows

Most long-history stations have multi-year early gaps (sparse campaign era), so
the default `max_gap_years=0.5` gate rejects them: only ~20% of the network
passes untouched. The fix is a per-station **fit window** in
`gps-config-data/analysis-lane/fit_windows.csv`
(`sta,window_start,window_end,max_gap_years,min_epochs,min_span_years,steps,comment`)
that fits the stored-detrend trajectory on a recent continuous **pre-unrest**
window. Edit that catalog, re-run `setup-testcfg.sh`, then
`ESTIMATE=1 ./plot-views.sh <STA>`.
