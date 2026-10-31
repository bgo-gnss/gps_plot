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
and view (they default to plate-periodic + cleaned; `ts` follows the plot). `--rate-segments steps|events` = one rate per span between events (design doc). `--view cleaned` detects outliers with the prior-free running window by default (gps_analysis design §15; `--outlier-sigma K`, default 4.5; σ not in the file name); `--outlier-method model` (or `--stages`) = the old trajectory-fit detector, file gets `-modeldetect`. Antenna obstruction, two indices: icing days (GAMIT autcln index, `gps-autcln-harvest`; all components, light blue) and snow-burial days (geo_dataread.snow_burial; Up only, pale lilac) — each `--icing`/`--snow remove|keep|off` (default remove) and `--show-icing`/`--show-snow [N]` (independent of `--show-outliers`); `--obstruction` / `--show-obstruction [N]` set both (an explicit per-kind flag wins); `--icing-ratio R`. Network days (whole network solution bad that day; `gps-network-days` list in pos_dir): `--network remove|keep|off` (default remove, every component), `--show-network [N]` draws them dark grey (`--operational` shows all). `--common-mode subtract|off` (default subtract): the daily network common mode (`network_cm.itrf14.csv`) is subtracted before detection on okada 24 h itrf08/itrf14 series only; high-pass, so rates/seasonals/records are unchanged; note "network common mode removed".
**Production batch contract (2026-10-21):** `gpsplot ts active` = the production set (stations.cfg, not passive/discontinued/inactive, in the box, with a series); one line per failure + a summary, exit 1 if any plot failed (no data in the period = skip, not failure); `--strict` refuses degraded output (raw under -cleaned/-detrend, no plate) before writing; `--manifest FILE` lists the files written (publish these, never glob); `--legacy-names` drops the frame tag (`STA-plate-90d.png`); `--save` alone = png. `--quake TIME,LAT,LON,MAG[,NAME]` marks an earthquake (layers/quakes.py). **`gpsplot run [STA ...]`** expands `plots.yaml` (`products.py`; stations narrow the selected jobs; baselines from station_areas.yaml `baselines:` (pairs), stacks from its `stacks:` (groups), plot roots from its `category_roots:`; design + `args:`/`prepare:` in `docs/plots-yaml-design.md`; `--update-pos [SCRIPT]` first runs plots.yaml `update_pos:` (the laptop's update_pos.sh), only when flagged, a failure warns and plots the series as they are).
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
plots.yaml `tool: map` products run maps through `gpsplot run` (Svartsengi
year + 3 months on one map, `--min-fill 0.6` per window, vertical arrows of
the windows side by side; `--last FROM:TO` = a window FROM to TO days back,
e.g. 3 + 3 months; `ts_stations:` adds check series = `ts --rate-last`
with the map's windows and ref, the arrows' numbers; stale map files are
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
(`--track-path-days`).
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
