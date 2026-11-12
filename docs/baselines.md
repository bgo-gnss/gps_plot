# Baselines: sign, axes, directions, polar (2026-10-31)

How gpsplot orients a baseline A-B (rover A, reference B) — daily (`ts`,
`stack`) and RTK (`rtk`) alike — and the two tools built on it, `gpsplot
direction` and `gpsplot polar`. Design and decisions: the plan
`.plan-baseline-frames.md` and destination `.interrogate-baseline-frames.md`
at the gpslibrary root.

## The sign: + = lengthening (`--sign apart`, default)

Each of N/E/U is signed per pair so that **+ is the baseline lengthening in
that component**: the a-priori vector B→A (station_coord.xyz,
`geo_dataread.station_catalog.apriori_baseline`) decides each sign
(`geofunc.axes.sign_axes`). A component whose a-priori length is under 1 m
stays raw (A − B): no direction to lengthen in. The labels say it
(`labels.py`, is/en). `--sign raw` gives the old plain A − B.

**This changed every published baseline plot under unchanged file names**
(2026-10-30). A component that used to go down when the stations moved apart
may now go up. Users of the plots got a note (below).

## Rotated axes: `--axes neu|along|3d|motion|slope`

Any non-neu mode adds `-<axes>` to the file name and rotates values, σ and
the full covariance (RTK series carry the N/E/U correlations, `Series.corr`;
`geo_dataread.rtk_read` v2 cache) with `geofunc.axes`; rows are the new axes
in N/E/U.

| mode | axes | direction from |
|------|------|----------------|
| `neu` | North / East / Up, signed by `--sign` | a-priori vector |
| `along` | along / across / up: horizontal along B→A (+ = lengthening), across 90° clockwise, Up as in neu | a-priori; the data vector (RTK level, newest common .pos epoch) if > 0.5° off |
| `3d` | along the 3-D line between the antennas / across / normal | as `along` |
| `motion` | motion / across / up toward the pair's motion azimuth | `directions:` motion |
| `slope` | downslope / across / normal of the pair's slope plane | `directions:` slope |

`motion`/`slope` take their direction, in order, from `--direction AZ` /
`--slope AZ,DIP`, `--direction-from START:END` (estimated as `gpsplot
direction`), else station_areas.yaml. A pair without one is dropped (stack
drops lone stations for any non-neu mode). Overlays (network days, icing,
snow burial, outliers) on a mixing rotation: a day set aside in any component
is set aside in all, by priority netday > iced > buried > outliers.

Code: `axes.py` (`orient`, `pair_frame`, `pair_orient`, `Orienter` for RTK,
`figure_labels`), `sources/gps.py::fetch` (daily pairs signed there), `rtk.py`.
`orient` refuses a series already oriented (no double signing).

## station_areas.yaml `directions:`

Per area, the fixed axes of `--axes motion|slope`:

```yaml
directions:
  SVIN-SKFC: {motion: 292, slope: [307, 23]}   # azimuth; [downslope azimuth, dip]
  SVIE-SKFC: {motion: 334, slope: [260, 21]}
```

Read by `direction.config_directions`; `products.area_directions` gives a
`gpsplot run` job only the pairs that have the direction its run needs.

## `gpsplot direction` — the line to put there

```bash
gpsplot direction SVIN-SKFC SVIE-SKFC -s 20260920 -e 20261028 --slope-from-dem
gpsplot direction SEY1-SEYD --rtk -s 20261029-12:00 -e 20261029-18:00 --window 30min
```

Robust end levels (median over `--window` at each end; σ = max(formal,
1.2533·1.4826·MAD/√n)) → azimuth, plunge (+ down) and significance
(|horizontal| > 3σ), from `gps_analysis.direction.motion_direction`. An
insignificant pair gets no line (SVIN-SVIE, 2026-10). A baseline here is
rover − reference, unsigned: where the rover went. `--slope-from-dem` adds
the DEM plane at the rover (see Terrain); the note names the DEM used.

## `gpsplot polar` — where the rovers went

```bash
gpsplot polar SVIN-SKFC SVIE-SKFC --together --special 90d --save png
gpsplot polar SVIE-SKFC --onset 20260920 -e 20261029 --save pdf
gpsplot polar SVIN-SKFC SVIE-SKFC --together -s 20260923 --rtk --save png
gpsplot polar SVIN-SVIE -s 20260923 --rtk --rtk-rate 72h --save png
```

The horizontal displacement since an onset, coloured by time. `--onset auto`
(default): the newest events.yaml event, of any kind, whose region holds one of
the rovers and whose onset is inside the period (the footnote names it), else
the start of the period; also a date, an event id, or `start`. Level = median of
the first `--window` after it. Panels (`--panels`, default `sector,map,profile`):

- **sector** — compass azimuth and size; motion direction (red), across
  (blue), DEM downslope. Its limits come from points ≥ 15 % of the largest;
  smaller points outside are drawn at the centre and counted, never hidden.
- **map** — the paths on the terrain around the rovers (grey hillshade lit
  from 315°/45°, ground steeper than 30° tinted orange, contours), each
  epoch a dot coloured by days since the onset (turbo, as the track maps),
  scaled up (title: m per mm; a dot scale bar) — the same convention as a
  velocity arrow on a map. BGÓ 2026-10-31, from the 2 m ArcticDEM dots
  experiment; the slope colour field and fall lines went.
  Dressed as the GMT maps (BGÓ 2026-11-08): tick labels inside the frame
  (`200 m E`, `400 m N`), a + at 0, 0 (the rovers' centre) with its
  coordinates top right, and an Iceland locator drawn as the GMT maps' (no
  frame or sea; the coast, fissure swarms, glaciers and central volcanoes of
  `insets.LOCATOR_*`, their colours too) with the map in red.
- **profile** (`profile.py`; default since BGÓ 2026-11-01, in place of the
  section) — one sub-panel per pair: the DEM ground along its motion,
  ±250 m through the rover (its own DEM window, wider than the map's), true
  heights with equal axes so the angles are true, all sub-panels one width
  and their heights following the ground. The rover stands on the ground;
  its daily points hang from it scaled up (m per mm, the same in both
  directions; a 10 mm-ish dot bar). Drawn with it:
  - the 50 m plane's apparent dip along the motion (dashed);
  - weekly medians (diamonds, a week needs ≥ 3 days);
  - the **plunge**: a line through the rover fitted to the daily points
    with their σ, errors in both directions (effective variance), with a 1σ
    wedge, scaled up by the misfit only when χ²/dof > 1. Days are not
    independent, so read it as a rough interval.
  The daily points are reduced to daily medians first, so weeks and the
  fit count days (the `--rtk` tail is drawn over them, never fitted). A westward motion runs to the left, as in the section. Without a
  DEM the heights are from the rover. `--panels profile` alone gives a
  profile-only figure (13 in wide, the colour bar in the first sub-panel).
- **section** — (optional) along the motion against Up [mm], with the
  terrain's apparent dip. A westward motion runs to the left (seen from the
  south), the compass points at the axis ends (BGÓ 2026-10-31).
- **hodograph** — north against east (optional; redundant with the sector).

### `--rtk`: the RTK on top of the daily figure (`polar_rtk.py`)

The figure is the daily one, unchanged (daily points, T line, 14-day
arrow); `--rtk` adds per pair (BGÓ 2026-11-12):

- **the RTK tail**: the newest `--rtk-days` (default 3d) of FIXED RTK
  solutions (RTKLIB Q 2 float and 5 single dropped), `--rtk-interval` 1h
  (default) or 10min medians, small dots coloured by time as the daily ones,
  with their 12 h running mean as a line in the pair's RTK colour (maroon,
  indigo, …), in the sector, map and profile. Moved onto the daily path:
  the median, over up to 14 common days, of (daily point − the RTK's own UTC
  day median, days with bins over half the day); the log prints the shift
  and the day-median scatter. Every daily statistic (T line, direction,
  sector limits, dot scale, weekly medians, plunge) stays on the daily
  points.
- **`--rate both|daily|rtk`** (BGÓ 2026-11-12: the RTK arrow dwarfed the
  daily one): `both` (default) draws both arrows; `daily` only the 14-day
  rate of the daily solutions, `rtk` only the RTK rate (the sector's and
  profile's 14-day step then follows the RTK rate, when it passes its test).
  The black arrow is the average since the onset (daily, above 2 σ); on
  the RTK figure with `--rtk-since DATE` the RTK average since the newest
  episode's onset (BGÓ 2026-11-12: 8 Nov 00:00 UTC at Svínafellsheiði).
  Files `-rate-daily` / `-rate-rtk`. The `--rate rtk` run also writes
  `<pairs>-polar-<period>-rtk-rates.json`: per rover each arrow's window,
  end and whether it is drawn, so `gpsplot rtk-tie --rates` draws the same
  window's rate as a line on the standard time series (fitted there, in
  its own frame). The Vöktun page shows the two figures side by side, each
  over the rtk-tie figure from 1 Aug (`--since 2026-08-01`): the daily
  frame figure under the daily one, the RTK + daily under the RTK one.
- **the RTK rate**, a line per component over the newest 10 and 30 min
  (1-minute values) and 6, 12, 24 and 72 h (`--rtk-rate`, default
  `10min,30min,6h,12h,24h,72h`). The SHORTEST window that passes its test
  is drawn on the map as an arrow in the RTK colour with its 1-σ ellipse
  (no wedge), at the 14-day arrows' scale (BGÓ 2026-11-12): so a slow,
  long-running signal shows as the 72 h arrow until a sudden speed-up
  takes over. Tests:
  - the longest window: its speed above 2 σ (along it);
  - the shorter ones: their CHANGE d = v − v_bg from the rate over the
    72 h before the window, z = |d/σ| (N, E; σ = 1.4826·MAD of d at every
    end over the last 7 d), above max(floor, quantile of z over those
    7 d): under an hour floor 4 and the 99.9 %; 6 h 3, 12 h 3.5, 24 h 4,
    with the 99 % (seven days hold only 14 / 7 independent 12 h / 24 h
    windows, so their quantile is a poor guide).
    The speed itself against 2 σ fired nearly every hour (a 2-D speed
    passes 2 σ in exp(−2) ≈ 13 % of noise; six windows compound it).
  The minute windows use the 1-minute values cleaned (`clean_minutes`):
  isolated spikes out by a CENTRED 5-minute Hampel (5 robust σ, ≥ 2 mm; a
  ramp passes, however fast — rtk_accel's trailing Hampel would cut a
  collapse's first hour), then minus the sidereal multipath model (the
  median of the high-passed values 1…7 sidereal days before, ≥ 3).
  Nothing passes: no arrow, a legend line "RTK no speed-up; 72 h v ± σ
  mm/d" (short: the map's legend sits beside its colour bar, which moves
  to the top of the map when the legend still reaches it); an arrow's line names the window, the speed and, for a
  shorter window, "speed-up z > limit". Up is in the log only, which lists
  every window. σ of a rate is
  empirical, the larger per component of (a) the residual scatter's
  white-noise σ of the slope × the block-mean scale k of the residuals
  (blocks of window/8) and (b) the repeatability: the window rate for
  every end over max(7 d, 4 windows), its robust scatter (1.4826·MAD)
  about a line through them in time. (b) catches the sidereal multipath
  and day-to-day atmosphere, which (a) cannot see in one window.
  Measured 2026-11-12 (hourly replays over 4 days, 8–12 Nov): the minute
  and 6 h windows never fired; 24 h on SVIN-SVIE 3 times (5 mm/d against
  ~2 on 8 Nov, the new movement); 12 h on SVIE-SKFC once (70 mm/d, noise:
  the 1 % by design). The change a window needs: SVIN-SVIE 10 min ≈ 6 m/d,
  30 min ≈ 5 m/d, 6 h ≈ 40, 12 h ≈ 15, 24 h ≈ 7 mm/d; the SKFC pairs:
  10 min 25–32 m/d, 30 min 7–8 m/d, 6 h ≈ 200, 12 h ≈ 75, 24 h ≈ 35 mm/d.
  Cleaning halved the SVIN-SVIE 10 min limit (12 → 6 m/d) and quartered
  SVIE-SKFC's; the 1-minute tails stay heavy (the 99.9 % of z is 10–45 σ):
  the minute windows see a collapse only.

Status line: the daily one plus `RTK to <dd Mon HH:MM> UTC`. Measured
2026-11-12: SKFC pairs 24 h σ 6–10 mm/d horizontal (never significant at
~1–4 mm/d motion), 72 h 1.2–1.6 mm/d; SVIN-SVIE 24 h 1.3–2 mm/d, 72 h
0.4–0.7. Day-median alignment scatter N/E 2–3.5 mm (SKFC pairs), 0.5 mm
(SVIN-SVIE); Up 19 mm (SKFC pairs): their RTK Up tail is not to be read.
Before 2026-11-12 `--rtk` drew the RTK 10-min medians (float included)
alone, from their first epoch.

The colour bar (days since the onset) sits inside the map, else inside the
profile, as on the other maps; only a figure with neither has it on the side.
The colours span only the newest `--colour-days` (default 28, BGÓ
2026-11-08); older points, in every panel, are open: a black outline
without fill. `--colour-days 0` colours them all. The colour bar then runs
−N … 0 days before the newest day ("last 28 days (days before 06.11.2026)"),
starts with an open triangle, and an open circle under it says "since
23.09.2026 until 28 days ago" (as the track maps, BGÓ 2026-11-08). The panels beside the map sit 0.08 in from it (their
labels), the figure trimmed by the width that frees.
Title: `A, B -- C movement since <onset>` (BGÓ 2026-11-02). No footnote
under or in the figure; `gpsplot polar` prints it (series, onset, motion
directions, DEM) to the run log instead.

`--together` draws all pairs in one figure, one marker per pair (the plots.yaml
`polar` product does this per `stacks:` group). Files: `A_B-polar-<period>`
(`-rtk`); together, the rovers then the references: `SVIN_SVIE_SKFC-polar-90d`.
A fixed start (`-s`, plots.yaml `fixedstart:`) without `-e` is
`since<YYYYMMDD>`, one file redrawn daily: the Svínafellsheiði product is
`SVIN_SVIE_SKFC-polar-since20260923` (BGÓ 2026-11-03; it was 90d + year).
The stacked figure (map + column) is framed on its map (`_frame_on_map`): no
map title, the figure title and status line just over the map (centred on
the figure), the logo
in its top-left corner on a white box, the column fitted to the map's height
(sector title on its top edge, lowest profile frame on its bottom edge); the
map column is as wide as the figure is high (`_square_map_ratio`). Other
layouts keep the title on top, the logo left of it.

## Terrain (`terrain.py`)

Slope at a station = a plane fitted to the DEM within `--slope-radius`
(default 50 m) of it → downslope azimuth, dip, rms. DEM choice
(`--dem-res auto|2m|5m`, `terrain.pick_dem`): a **2 m area grid** under
`gis_dir/dem/arcticdem_2m/*.nc` that holds the whole window, else the 5 m
ISN93 DEM (`dem/layers/iceland_land_5m_isn93.tif`).

The 2 m grids (2026-10-30): `svinafellsheidi.nc`, `seydisfjordur.nc`, cut
with `pygmt.grdcut` from ArcticDEM v-1 (LMÍ prerelease Aug 2019, ISN93,
**ellipsoidal**) tiles on the gogn drive
(`nas/luk/gogn/haedarlikon/arcticdem/lmi-prerelease-august-2019/dems/
Iceland_v-1_ArcticDEM_Mosaic-tile-NN-median.tif`; tile 15 Svínafellsheiði,
11 Seyðisfjörður), with the area's mean geoid (`dem/layers/geoid_500m_filled.tif`)
subtracted. A new area: same cut, the stations' box plus ~2 km.
Reykjanes (2026-10-31): 14 grids `reykjanes-<stations>.nc`, ±1.5 km around
each station of the area, overlapping boxes merged (one 29 × 15 km grid holds
Svartsengi, Fagradalsfjall, Krýsuvík; 347 MB in all), from tiles 12/13/18/19
through a VRT; recipe `gis/dem/arcticdem_2m_reykjanes.py`. They predate the
2021– lavas (BGÓ: fine, the lava outlines are overlaid); heights agree with the
5 m DEM to 0.1 m. No grid at ELEY (Eldey: no data in either DEM). Only `.nc` /
`.tif` files in the folder are read. GMT gotcha: it took a temporary
`reykjanes-undh.tif` for a GTOPO30 tile by its name — read such a file as
`name.tif=gd`. The `directions:` slopes in station_areas.yaml
were measured on the 5 m DEM; at SVIE the 2 m plane agrees (260°/21°).

## plots.yaml

- Runs and products take `areas: [...]` to limit them to those areas.
- baselines: `--axes along` 90d+year for svinafellsheidi, seydisfjordur,
  reykjanes; `--axes motion` for svinafellsheidi (`-along` / `-motion` files).
- `polar` product: `tool: polar`, `stations: area_stacks` (→ `--together`),
  svinafellsheidi, 90d + year.
- `rtk:` run `7d-along` (`--longterm week --rtk-overlap 2 --axes along`, same
  three areas), in the long timer (`gpsplot rtk-run --run 4d,7d,14d,7d-along`).
- aflogun (`graphs.py`) lists the `-along`/`-motion` baseline variants and
  the polar files under each station in their name: "A-B · polar-<variant>",
  together "SVIN-SVIE-SKFC · polar-90d" (aflogun !13).

## Daily baselines: layering (2026-10-08)

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

## Note to plot users (the sign change)

> **Breytt formerki á grunnlínum / baseline sign change (2026-10-30).**
> Á grunnlínuritum (A-B) þýðir + nú að grunnlínan lengist í þeim þætti
> (N/A/U), þ.e. stöðvarnar fjarlægjast. Áður var sýnt A − B beint, svo sumir
> þættir snúast við þótt skráarnöfn séu óbreytt. Ný rit með `-along` sýna
> lengingu eftir grunnlínunni og `-motion` hreyfingu í hreyfistefnu.
>
> On baseline plots (A-B), + now means the baseline lengthens in that
> component (N/E/U): the stations move apart. Before, A − B was plotted as is,
> so some components are flipped although the file names are unchanged. New
> `-along` plots show lengthening along the baseline, `-motion` plots the
> motion in its own direction.
