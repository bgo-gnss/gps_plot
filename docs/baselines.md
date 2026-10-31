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
gpsplot polar SVIE-SKFC --rtk --special week --save png
```

The horizontal displacement since an onset (`--onset` date or events.yaml id,
default the start of the period; level = median of the first `--window`),
coloured by time. Panels (`--panels`, default `sector,map,section`):

- **sector** — compass azimuth and size; motion direction (red), across
  (blue), DEM downslope. Its limits come from points ≥ 15 % of the largest;
  smaller points outside are drawn at the centre and counted, never hidden.
- **map** — the paths on the terrain around the rovers (slope colour 0–45°,
  hillshade, contours, fall lines), scaled up (title: m per mm) — the same
  convention as a velocity arrow on a map.
- **section** — along the motion against Up, with the terrain's apparent dip.
- **hodograph** — north against east (optional; redundant with the sector).

`--together` draws all pairs in one figure, one marker per pair (the plots.yaml
`polar` product does this per `stacks:` group). Files: `A_B-polar-<period>`
(`-rtk`); together, the rovers then the references: `SVIN_SVIE_SKFC-polar-90d`.

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
subtracted. A new area: same cut, the stations' box plus ~2 km. No Reykjanes
grid: 2019 predates the lavas. The `directions:` slopes in station_areas.yaml
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
  two-station polar files ("A-B · polar-<variant>"). **Follow-up:** its
  `_POLAR_PNG` matches only `A_B-polar-*`, so the together figures the
  `polar` product writes (`SVIN_SVIE_SKFC-polar-*`) are not listed yet.

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
