# gpsplot subcommands — options and behaviour

Routed out of `gps_plot/CLAUDE.md` on 2026-10-31 (over its size budget),
**moved verbatim**. The subcommand design and staged plan are in
`cli-subcommands-design.md`; baselines in `baselines.md`; maps in
`map-lane.md` and `map-config-guide.md`; plots.yaml in `plots-yaml-design.md`.

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
and view (they default to plate-periodic + cleaned; `ts` follows the plot). `--rate-segments steps|events|auto` = one rate per span between events (design doc); `auto` = between the rate changes found in the plotted series (`gps_analysis.breaks.rate_breaks`: weekly medians, continuous hinge model, month-grid candidates ≥ 0.5 yr apart (2/12 for spans ≤ 3 yr), greedy + refinement, a break kept while the BIC drops ≥ `--break-gain` (default 25)); searched on the `--along` first component, else on the horizontals along their own motion direction, so all panels share the breaks; drawn as brown lines on every panel, the breaks and BIC gains printed; combines with steps/events (PLAN-ts-quakes Q-T3). `--one-panel` = the first panel alone, 13 × 5.33 in (with `--along` the motion toward the azimuth; rates, breaks and `--quakes` on it, the quake legend under the date axis; notes of the dropped panels go with them); file `-1panel` (Q-T7). The Bárðarbunga figure: `gpsplot ts VONC -s 20180101 --along opt --rate-ref 20130901:20140131 --rate-segments auto --quakes --one-panel`. `--rate-ref A:B` = the series relative to the product rate of window A:B: v·(t − t0) subtracted per component (zero kept at the first epoch), the window and its N/E/U rate in the title ("miðað við hraða …"), every `--rate` line reported relative to it (σ in quadrature; the reference window itself keeps its own σ); refused with `--model`, daily .pos only; files `-relrate` (PLAN-ts-quakes Q-T4; e.g. VONC `--rate-ref 20130901:20140131`). **Seasonal removed by default:** `--rate-ref` or `--along` without `--ref`/`--detrend-terms`/`--model` reads `--ref detrend --detrend-terms periodic` (the product's plate-periodic; the stored LINE stays in, it can be fitted to too short a quiet span, e.g. VONC before the Bárðarbunga signal; BGÓ 2026-11-16); an explicit `--ref plate` with a `--rate-ref` window shorter than a year warns, because the reference rate then takes the seasonal's slope (VONC: N 2.7 instead of 5.4 mm/yr, the direction 9° off). `--along opt|motion|AZ` = one station as motion/across/up toward an azimuth (`axes.station_frame`): `opt` = the direction of the plotted motion by `gps_analysis.direction.motion_direction` (the estimator of `gpsplot direction`, 30-day end levels (at most half the plotted span), after `--rate-ref`), `motion` = station_areas.yaml `directions:`, a number = itself; azimuth in the title ("besta stefna 284.6° ±0.5°"), `--rate` lines rotated with it (σ through R·C·Rᵀ with the record's N–E correlation); single stations only (baselines: `--axes motion`), refused with `--model`; files `-along[-opt|-motion]` (PLAN-ts-quakes Q-T5). `--quakes [REGION]` = catalogue earthquakes on the top panel (`layers/quakes.py`, PLAN-ts-quakes Q-T6): `geo_dataread.seismic.catalogue_for` over the panel's x range (SIL files before 2026-02-01, SeisComP Quakes API after, a hard cut) + `moment_estimate`; magnitudes on a right axis (stem + dot; grey < 4, blue 4–`--quake-star` (4.5), red star above, value written from `--quake-label` (5); automatic events hollow), the cumulative moment on an outer right axis (grey fill behind the data, black step in front, dashed after the seam = estimated from SeisComP magnitudes), the seam a red dashed line; the legend title counts the automatic events and their share of the moment (≥ 0.5 %); `--no-moment` = magnitudes only. REGION = a name in quake_regions.yaml (`--quake-regions FILE`, default the deployed one), W,S,E,N or LAT,LON,KM; none = the station area's `default:` region (refused when the station's areas name none, or several). The selection/catalogue options are geo_dataread's `gpsquake` ones with a `quake-` prefix (`gpsquake.add_quake_args`): `--quake-source auto|sil|seiscomp|fdsn`, `--quake-mode catalog|automatic|reviewed`, `--quake-min-q Q` (default 10; raise it in swarms, 0 = all), `--quake-no-duplicates`, `--quake-min/max-mag`, `--quake-min/max-depth`, `--quake-types`, `--quake-polygon`, `--quake-no-cache`. One console line per station (region, n, automatic, moment, parts). The panel's right edge moves in to 0.84 for the two axes; files `-quakes` (the region is in the legend, not the name). `--view cleaned` detects outliers with the prior-free running window by default (gps_analysis design §15; `--outlier-sigma K`, default 4.5; σ not in the file name); `--outlier-method model` (or `--stages`) = the old trajectory-fit detector, file gets `-modeldetect`. Antenna obstruction, two indices: icing days (GAMIT autcln index, `gps-autcln-harvest`; all components, light blue) and snow-burial days (geo_dataread.snow_burial; Up only, pale lilac) — each `--icing`/`--snow remove|keep|off` (default remove) and `--show-icing`/`--show-snow [N]` (independent of `--show-outliers`); `--obstruction` / `--show-obstruction [N]` set both (an explicit per-kind flag wins); `--icing-ratio R`. Network days (whole network solution bad that day; `gps-network-days` list in pos_dir): `--network remove|keep|off` (default remove, every component), `--show-network [N]` draws them dark grey (`--operational` shows all). `--common-mode subtract|off` (default subtract): the daily network common mode (`network_cm.itrf14.csv`) is subtracted before detection on okada 24 h itrf08/itrf14 series only; high-pass, so rates/seasonals/records are unchanged; note "network common mode removed". `--uncert` (the read-time formal-σ screen) defaults to 15 mm (was 10; BGÓ 2026-11-22: VONC's snowy days went missing); a kept day with σ above 10 mm in any component gets bold orange error bars on all three panels and a count on the East panel ("σ above 10 mm: N days (orange bars)", `timesmatplt.mark_high_sigma`). Such days are kept only among the last `--high-sigma-days N` days (default 30; 0 = never, negative = always), older ones removed as the 10 mm screen did (`drop_old_high_sigma`).
**Production batch contract (2026-10-21):** `gpsplot ts active` = the production set (stations.cfg, not passive/discontinued/inactive, in the box, with a series); one line per failure + a summary, exit 1 if any plot failed (no data in the period = skip, not failure); `--strict` refuses degraded output (raw under -cleaned/-detrend, no plate) before writing; `--manifest FILE` lists the files written (publish these, never glob); `--legacy-names` drops the frame tag (`STA-plate-90d.png`); `--save` alone = png. `--quake TIME,LAT,LON,MAG[,NAME]` marks an earthquake (layers/quakes.py). **`gpsplot run [STA ...]`** expands `plots.yaml` (`products.py`; stations narrow the selected jobs; baselines from station_areas.yaml `baselines:` (pairs), stacks from its `stacks:` (groups), plot roots from its `category_roots:`; design + `args:`/`prepare:` in `docs/plots-yaml-design.md`; `--update-pos [SCRIPT]` first runs plots.yaml `update_pos:` (the laptop's update_pos.sh), only when flagged, a failure warns and plots the series as they are).
`gpsplot ts all` = every station with a .pos series of the frame in the Iceland box (geo_dataread `pos_stations`; TOT: stations.cfg); `--region=W/E/S/N` (with `=`, west is negative) replaces the box and filters lists, `--with-record` keeps stations with a detrend record.
`gpsplot stack "SEY?-SEYD" HOFN --offset 20` overlays stations and baselines
(patterns match stations.cfg). `--events YYYYMMDD[,COLOR] …` / `--eventf FILE` draw event lines on every
panel in `ts`, `stack` and `rtk` alike (`cli/common.add_event_options`; not
`seasonal`, which folds onto one year). A late-starting stack series is backcast
from its own data before the window when it has some (read a year back:
level and rate before the gap, `stack._from_history`; SVIN 2026, whose
in-window rate was the landslide's), else along a robust line through its
first data (MAD-clipped, `stack._rate_line`). `--zero-before YYYYMMDD|event`
zeroes each stack series on its median before that date (`event`: the newest
`--events` line in the window; at least 2 epochs) instead of on its first,
noisy, epoch — the Svínafellsheiði stacks, BGÓ 2026-11-02. Baselines to one
reference are titled `A, B -- REF baselines`. `ts`, `stack` and `rtk`
carry the Veðurstofa logo top left of the North panel, `polar` in the band
left of its title (`tplt.inpLogo`,
`assets/IMO-Logo-IS-01-Blue.png`); the stack/rtk legend sits at the foot of
the Up panel (BGÓ 2026-11-02). And a detrended series is
zeroed on its first clean samples (`sources.gps.first_clean`): one undetected
outlier at the start no longer sets the zero (OLAC, July 2021). `gpsplot seasonal NYLA` folds a station's seasonal onto one year: data, own, regional LOO (design doc). `gpsplot rtk "SEY?-SEYD" [--daily]` plots the
RTK baselines of rtk.vedur.is (+ the daily GLOBK ones). `gpsplot rtk-map
--pairs A-B,... --stations S,... --region W/E/S/N` draws the last `--hours`
(12) of RTK motion on a map: each pair's change (last 60 min median minus
first), the pairs solved as one network in its own frame (no net motion of
its stations; `--anchors` holds stations at zero instead, but far anchors'
long float links trended 20-50 mm in 12 h, BGÓ 2026-11-02), vertical in the
labels; the operational Reykjanes map every 5 min (`rtk_map.py`). `gpsplot map velocity
--region W/E/S/N --window A:B --window C:D` draws windowed velocity fits per
window (BGÓ's Reykjanes maps); also `station`, `deformation` (`--vertical`,
`--values STA,…|all` at the arrow tips, `--late-start`; docs/map-lane.md), `slip`.
plots.yaml `tool: ts` products with `area:` and named runs (`name:`, `caption:`) put one
station's figure beside the area's maps (Bárðarbunga VONC + quakes; design doc "Named ts
figures"). plots.yaml `tool: map` products run maps through `gpsplot run` (Svartsengi
year + 3 months on one map, `--min-fill 0.6` per window, vertical arrows of
the windows side by side; `--last FROM:TO` = a window FROM to TO days back,
e.g. 3 + 3 months; `ts_stations:` adds check series = `ts --rate-last`
with the map's windows and ref, the arrows' numbers; stale map files (no run writes them, unmodified for 2 days) are
deleted; design doc "Map products"). `map station --track EVENT`
= daily positions since an event as time-coloured dots + rate arrows
(`track.py`; map-lane doc); colour bar small in the upper right;
`--track-series STA,…` (plots.yaml `ts_stations:` on a track map) draws each
track as a N/E series with the legend's rates on the data, `<dir>/ts/`;
`--track-ref STA` = every track relative to STA (a baseline: shared motion
drops out); `--track-ref A,B,C…` = relative to their weighted mean (each detrended on
the pre-onset window first, days ≥ half of them have; glorg-like translation), a
reference station that moves since the onset (> 5 σ from the median rate, while
≥ 3 are left) is left out (`track.screen_references`; EYVI 2026-10);
days are matched by calendar `track.day_key`; the dotted path follows a 5-day running median
(`--track-path-days`). `--track-status` also writes `<figure>.status.json`
(`track_status.py`, schema 1, rates mm/yr): per station its role (station /
reference / dropped_reference), newest day, the rate over the last
`--track-status-days` days (default `--track-recent`; widened like the recent
rate for sparse days) and since the onset, each `[value, σ]` with the same σ
as the legend, and the newest day's movement (newest − previous day, σ = √2 ×
the pre-onset daily scatter, significant > 3 σ); it goes into the manifest
with the map. aflogun's *Vöktun* tab reads it (area `monitoring:` in
station_areas.yaml).
A monitoring look (BGÓ 2026-11-08, Svínafellsheiði): `--track-stations
SVIE,SVIN` draws arrows and legend lines for those stations only (the others
dots only, still references); `--track-colours recent` draws the recent rate
as the station-coloured arrow, only above 2 σ (its legend line then too, all
rate lines first; "no arrow: no motion above the noise" otherwise), and the
displacement since the onset as a solid grey T line with its ellipse at the
dots' scale (a T icon "10 mm movement since …", the numbers in grey);
`--track-colour-days 14` colours the dots of the last 14 days only (the
colour bar −14 … 0 days before the newest), older ones open black ("since
<onset> until 14 days ago"), as `gpsplot polar`.
BGÓ 2026-11-17: in that look the rates are in mm/d (the status file stays
mm/yr); the mean velocity since the onset is a black arrow (above 2 σ) over
the grey T line and under the station-coloured 14-day arrow, so it shows past
that arrow's tip when the motion slows below the mean; its numbers are one
black legend line ("since 23.09.: SVIE … mm/d to 329°, SVIN …"). The arrow
scale follows the arrows drawn (the longest, Up included, `ARROW_CM` / 1.5 cm
in the zoom inset) and shrinks where an arrow with its 1-σ ellipse would leave
the map or the inset (`track.arrow_scale_of`, as polar). The legend sits 0.1 cm
above the bottom annotations, a bottom zoom inset 0.05 cm on it.
Area layers for any map: `--layers volcanic` (FutureVolc swarms, central
volcanoes named, caldera rims) and `rural` (IS 50V main + local roads,
building points, farm names; `vector/build_09_is50v.sh`); overlapping
place names are dropped (`maps.apart`, first wins), station labels keep
off the logo. The layer catalogue is config: `map_layers.yaml` (gpsconfig
dir, or `--layers-file`) adds/restyles layers and presets over the built-in
one (`gis_layers.load_catalogue`), each layer a `--gmt` step with its own
`gmt:` args; plots.yaml map products take `layers:`, `gmt:`, `gmt_config:`.
Fallbacks: a missing layer file is skipped with a warning (`--strict-layers`
refuses), a missing Iceland DEM draws `@earth_relief_15s` (no SRTM north of
60°); `--layers list` marks missing files. Guide outline:
`docs/map-config-guide.md`.
Planned: seismic panels (on ice) — the staged plan and BGÓ's
decisions are in the design doc. Every step is gated by
`bin/check_cli_variants.py` (byte-identical PNGs vs a `main` export).

## Ad hoc figures (`--adhoc`, 2026-11-01)

One-off figures for an area or event, kept out of the production tree
(`adhoc.py`, hooked in the dispatcher, so every figure tool has it: ts,
stack, rtk, seasonal, polar, map; run/rtk-run/direction refuse it):

```
gpsplot polar SVIN-SKFC SVIE-SKFC --together --panels profile --adhoc --note "plunge vs. ground"
gpsplot ts SKFC --adhoc oraefajokull/jokulhlaup-2026 --save pdf
```

- Folder, area first: `<plots_dir>/locations/<root>/<area>/adhoc/<label>/`.
  `--adhoc [AREA][/LABEL]` (or `--adhoc=AREA/LABEL`): AREA a cdn slug of
  station_areas.yaml; LABEL a folder name, default today `YYYY-MM-DD` (the
  same command twice in a day overwrites; give a label to keep both).
- AREA left out: the area holding all the command's stations; when several
  do (SVIN/SVIE/SKFC are Öræfajökull members too), the one listing the
  command's baselines (`baselines:`/`stacks:`, either way round). Still
  ambiguous or none → refused with the candidates; give AREA.
- `-d`/`--figDir` with `--adhoc` is refused; no `--save` → `--save png`.
  A map's KIND comes before `--adhoc` (a bare lower-case word after it is
  read as the AREA).
- Beside every figure written: `<figure>.json` — the command as typed, argv,
  area, label, note, UTC time, user, gps_plot commit (`git describe`, else
  the installed version). Re-run = the `command`; a figure that became
  regular moves into plots.yaml.
- Out of production: `gpsplot run`'s stale sweep reads only its products'
  own folders (not subfolders), `publish.manifest` never lists `adhoc/`, so
  nothing is published to the CDN and aflogun does not show them.
