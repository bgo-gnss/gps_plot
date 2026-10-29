# RTK monitoring plots: gpsplot + aflogun (design, 2026-10-29)

Goal (BGÓ): watch RTK baselines in near real time AND against the long daily
GLOBK record, and in aflogun click a baseline drawn on the map to see its
plot.

## 1. The plot (done on `feat/rtk-longterm`)

`gpsplot rtk A-B --longterm 90d|year|full [--rtk-span week]`: the daily
GLOBK baseline is the anchor, the RTK tail (10-min medians) is moved onto it
by the median over their overlap; without overlap it keeps its own zero and
the figure says "NOT aligned" (BGÓ chose this over pinning or leaving RTK
out). Float solutions are in by default (BGÓ); `--fixed-only` drops them.
A pair processed only as B-A on rtk.vedur.is is read negated.

BGÓ 2026-10-29: RTK on a long axis makes no sense; the joined plot is for
intermediate spans (2–14 days), with RTK starting just before the last daily
point: `--longterm 14d --rtk-overlap 2`. Two products per pair: LONG = the
existing daily baseline plot (gpsplot run), SHORT = RTK + daily, 14 d. The
`--longterm 90d|year|full` CLI stays for occasional use.

## 2. The runner: `gpsplot rtk-run` (2026-10-29)

Separate from `gpsplot run` (daily cadence, prepare steps, stale pruning)
because RTK plots go stale in minutes. Module `gps_plot.rtk_run`.

- **Config:** an optional top-level `rtk:` section in plots.yaml (`gpsplot
  run` ignores it); without one the default is six runs (BGÓ): RTK alone over
  `6h`, `12h` and `2d` (`--special 6h|12h|twodays`, the rtk_gps spans), and
  `4d` (`--longterm 4d --rtk-overlap 1`), `7d` (`--longterm week
  --rtk-overlap 2`) and `14d` (`--longterm 14d --rtk-overlap 2`), into
  `locations/{root}/{cdn}/graphs/rtk`:

  ```yaml
  rtk:
    dir: locations/{root}/{cdn}/graphs/rtk
    runs:
      - { name: 6h, args: [--special 6h] }
      - { name: 12h, args: [--special 12h] }
      - { name: 2d, args: [--special twodays] }
      - { name: 4d, args: [--longterm 4d, --rtk-overlap 1] }
      - { name: 7d, args: [--longterm week, --rtk-overlap 2] }
      - { name: 14d, args: [--longterm 14d, --rtk-overlap 2] }
    overrides:
      svinafellsheidi: { args: [--rtk-span week] }
  ```
- **Pairs:** each area's `baselines:` + `stacks:` pairs (station_areas.yaml)
  that rtk.vedur.is processes either way round: 54 today (×6 runs).
- **Group figures:** each `stacks:` group with two or more RTK pairs is
  also drawn as one figure per run, as rtk_gps's multi-baseline plots:
  `<dir>/stack_<group>-rtk-<run>.png` (`stack_args:` / `overrides:
  {<area>/<group>: …}` add options). aflogun shows them under the
  *Grunnlínuhópar* button of Rauntímaplot.
- **Files:** `<dir>/A_B-rtk-<run>.png`. Names are stable and kept out of the
  `A_B-baseline-*` pattern that `gpsplot run` prunes. Each file is written in a
  temp dir beside its target and renamed over it. Stale `*-rtk-*` files in
  the rtk dirs are deleted (`--keep-stale` keeps them).
- **Lock:** `<plots_dir>/.rtk-run.lock` (flock). A run still going makes the
  next one exit 0.
- **Exit:** a pair with nothing to plot (stations down) is skipped and its
  old plot is kept. Exit 1 only when a plot raised.
- **Cost:** 54 plots take 144 s on a cold cache and ~40 s warm (only
  today's 1 Hz files are re-read), so a 5-min timer is fine.
- **Host:** the same host as `gpsplot run` (BGÓ). There is no CDN push yet:
  aflogun is tested locally against `plots_dir` while the server is awaited.

Timer (user systemd, 5 min):

```ini
# ~/.config/systemd/user/gpsplot-rtk.service
[Service]
Type=oneshot
ExecStart=%h/.miniforge/envs/gpslibrary/bin/gpsplot rtk-run
# ~/.config/systemd/user/gpsplot-rtk.timer
[Timer]
OnCalendar=*:0/5
Persistent=false
[Install]
WantedBy=timers.target
```

## 3. aflogun: click a baseline to show its plot

Today (only on the unmerged `feat/local-plots` → `feat/kort-captions`, MR !8)
`leafmap.baseline_lines` draws orange lines for the pairs of the shown
baseline/stack plots, hover-labelled `A-B`, not clickable. Plan, stacked on
!8 once it is merged:

- Each `dl.Polyline` gets a pattern id `{"type": "baseline-line", "pair":
  "A-B"}` and `n_clicks`, like the station markers. Under it goes a wider,
  transparent twin line, so near-parallel pairs (SVIN-SKFC and SVIE-SKFC)
  stay easy to hit.
- A `dcc.Store("baseline-pick")` holds the clicked pair. The callback reads
  `ctx.triggered_id` and ignores `n_clicks == 0`, because the layer is rebuilt
  from five inputs.
- The picked line is redrawn in a highlight colour. The static panel
  shows that pair's plots first, or only those: the daily baseline plots
  (long) and the RTK plot (short, §2 `A_B-rtk-<run>.png`, a new `rtk` frame
  matched by its own regex).
- The lines are drawn whenever the area has baselines, not only when a
  baseline frame is chosen, so they can be clicked.
- Clicking the same line again, or a station, clears the pick.
