# Map configuration guide — OUTLINE

*Status: outline only (BGÓ 2026-10-28: "not overreach creating the
documentation now"). Each bullet is a section to write; the facts it needs
are named so the writing is collection, not research.*

## 1. What controls a map (one page, the map of the config)

- The three files and who reads them: `plots.yaml` (which maps exist and
  what each draws), `map_layers.yaml` (what a layer is and how it looks),
  `postprocess.cfg [PATHS] gis_dir` (where the files are). All in
  gps-config-data → `~/.config/gpsconfig/`.
- Precedence, one diagram: built-in catalogue → `map_layers.yaml` →
  plots.yaml `defaults:` → product → run → command line (`--gmt` wins).

## 2. Discovering what is available (the CLI)

- `gpsplot map station --layers list` — every layer and preset, the
  catalogue it came from, `[missing: …]` for files not on this host.
- `gpsplot map --help` — `--gmt` lists the drawing steps (the layers are
  steps too); `--gmt-config` any GMT default.
- `gpsplot map --dem-cmap list` (palettes), `gpsplot run --dry-run` (what
  the daily run will draw, with every option spelled out).
- TO DECIDE: a `--gmt list` printing the steps with their PyGMT call and
  long-name/letter pairs (now only in `--help`).

## 3. Tutorial: a new area map, start to finish

- Pick region/projection by hand: `gpsplot map station … --region=W/E/S/N`.
- Add layers one at a time (`--layers volcanic`, then `rural`), restyle
  one (`--gmt calderas:pen=2p,red`), a GMT default (`--gmt-config
  MAP_FRAME_TYPE=plain`).
- Move it into plots.yaml: `tool: map`, `dir:` under the area,
  `layers:`, `gmt:`, `runs:` with `name:`; `gpsplot run --dry-run
  --product X`; deploy; it shows in aflogun Kort.
- A velocity map with check series (`ts_stations:`), a track map
  (`--track EVENT`, `--track-series`).

## 4. Tutorial: a new layer

- Get the data (WFS/ArcGIS query or a file), WGS84 GeoJSON or a
  `LON LAT NAME` text file, under `gis_dir`.
- Add it to `map_layers.yaml` (`file`, `help`, `pen`/`fill`/`symbol`,
  `label`/`font`, `gmt`); a preset if it belongs with others.
- If it comes from a service: a `build_NN_*.sh` step and an entry in
  `refresh_layers.sh` FILES/SCRIPT, so the production host refreshes it.

## 5. Reference

- `map_layers.yaml` keys (the header of the file is the short form).
- plots.yaml map product and run keys: `tool map dir stations args runs
  ts_stations layers gmt gmt_config`; run: `name args publish layers gmt
  gmt_config`; `defaults: save map_args map_layers map_gmt
  map_gmt_config`.
- Drawing steps and what each draws (gmt_overrides.LAYERS).

## 6. Data on the host: what is fetched, shipped, built

- Fetched (refresh_layers.sh, monthly gps-gis-refresh.timer): FutureVolc
  (07), Reykjanes LMÍ/NÍ/IMO (08), IS 50V (09); a failed or shrunken
  fetch keeps the old files.
- Built once (not fetched): coast/ice/water (05–06).
- Shipped: the Iceland DEM + its coast/glaciers/lakes, a private generic
  package `iceland-dem/<VERSION>` of gps-data on git.vedur.is (never
  GitHub): `gis/dem/publish_dem.sh`, `fetch_dem.sh`, `iceland-dem.sha256`;
  the DEM README's "Shipping" section.
- aflogun's interactive map: its own catalogue (`gislayers.json`) + the
  VÍ GIS registry (cached an hour, last good list on an outage).

## 7. When something is missing (fallbacks)

- A layer file missing: drawn without it, a warning; `--strict-layers`
  refuses instead.
- The Iceland DEM missing: GMT's `@earth_relief_15s` (SRTM 1″/3″ end at
  60°N), downloaded by GMT and cached in `~/.gmt` (needs the internet
  once).
- No IS 50V coast/lakes: GMT's own coast (GSHHG).
- A registry/service down: aflogun keeps the last list; the refresh keeps
  the old files.

## 8. Troubleshooting

- `--gmt` refused (unknown step/key: the message names the closest);
  the long-name/letter clash; a layer drawn but invisible (order:
  later on top); labels gone (`maps.apart`, inside the frame, the logo).

## Inline help to add to the config files

- `plots.yaml`: a header block on map products (the keys of §5, one line
  each) next to the existing standard/area comments.
- `map_layers.yaml`: done (header lists the keys); add one commented
  example of a new layer and of `gmt:`.
- `postprocess.cfg`: comment on `gis_dir`.
