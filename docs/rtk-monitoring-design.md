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

Open: a year axis squeezes a week of RTK into a few pixels. Next step is a
right-hand zoom column (last `--rtk-span`, RTK + daily) beside the long axis,
or two products per pair (long + short) shown side by side in aflogun.

## 2. The runner (next)

Separate from `gpsplot run` (daily cadence, stale pruning) because RTK
plots go stale in minutes:

- **Command:** `gpsplot run --only rtk` or its own `gpsplot rtk-run`, reading
  an `rtk:` product in plots.yaml. Pairs come from station_areas.yaml
  (`baselines:`/`stacks:`) that exist on rtk.vedur.is in either direction.
- **Host:** the same host as `gpsplot run` (BGÓ). It needs the read-only use
  of `/mnt_data/rtgpsdata` (never written), `pos_dir`, and the CDN push.
- **Cadence:** every 5–10 min from a systemd timer or cron, with `flock` so
  runs never overlap. Cost is small: finished days come from the
  `rtk_read` cache, only today's file is re-read.
- **Files:** `<area>/graphs/rtk/A_B-rtk-<span>.png`. These are stable names
  with no tuning parameters (plot-filenames rule). They are kept out of the
  `A_B-baseline-*` pattern so `gpsplot run`'s stale pruning never deletes
  them. Write to a temp file, then rename, so the CDN never serves half a PNG.
- **Publish:** its own rsync to the CDN after each run; the daily push is
  too slow.

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
  shows that pair's plots first, or only those: the baseline plot, the RTK
  long-term plot (§2 files, a new `rtk` frame matched by its own regex) and
  the RTK short plot.
- The lines are drawn whenever the area has baselines, not only when a
  baseline frame is chosen, so they can be clicked.
- Clicking the same line again, or a station, clears the pick.
