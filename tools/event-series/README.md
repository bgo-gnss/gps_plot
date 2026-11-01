# Event series: weekly velocity maps of an event, after the fact

`event_series.sh OUTDIR FIRST LAST [map velocity options]` draws one
`gpsplot map velocity` per week (ending FIRST, FIRST + 7, ... LAST) and a
contact sheet of them. It is for looking at an event afterwards; the
operational map is the daily `seydisfjordur_velocity` product of
plots.yaml (BGÓ 2026-11-01).

No `--scale`: each week picks its own (the 90th-percentile arrow 2 cm), so
an event week's 7-day arrows set the scale and a quiet week's 2-month ones
do. The legend arrow says what each map's scale is.

## Seyðisfjörður (SEY? relative to SEYD)

```bash
tools/event-series/event_series.sh ~/plots/sey-jan2025 2024-12-04 2025-03-05 \
  --region=-14.03/-13.965/65.2455/65.268 --dem iceland --layers seydisfjordur \
  --steep --sea-smooth 0.15 --locator-corner BL \
  --ref plate --min-fill 0.6 --last 61 --last 7 \
  --vector-ref SEYD --short-step --short-snr 2 --short-downslope \
  --short-coherent 2 --min-snr 1 --total
```

- blue: the 2 months before each frame's day; red: its last 7 days as a
  step against the 14 days before (`--short-step`), drawn only when it
  stands out of the noise, points downslope and 2 stations agree
- purple T line and number: how far, and which way, each station had
  moved since it was installed, up to that day (`--total`)
- `GPSPLOT="python -m gps_plot.cli"` (or a worktree's) runs another gpsplot

**Check a red week against the series.** A jump that goes back within
days is snow or ice on the antenna, not the slope: SEY3's 50 mm in the
week to 2025-01-29 was back to zero on 2025-02-02 (`gpsplot ts
SEY3-SEYD --show-obstruction`).
