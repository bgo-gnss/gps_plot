# Map configuration guide

*Draft, 2026-10-28. The map tools are still growing; this says how they
work today. Sections marked TODO wait for features.*

## 1. What controls a map

Three files in gps-config-data (deployed to `~/.config/gpsconfig/`):

| File | Says |
|---|---|
| `plots.yaml` | which maps the daily run makes, and what each draws |
| `map_layers.yaml` | what each layer is: its file, colour, label, symbol |
| `postprocess.cfg` `[PATHS] gis_dir` | where the GIS files are (`~/gps-data/gis`) |

A map's settings come in this order; a later one adds to or wins over an
earlier one:

1. gps_plot's built-in layer catalogue
2. `map_layers.yaml`
3. `plots.yaml` `defaults:` (`map_args`, `map_layers`, `map_gmt`, `map_gmt_config`)
4. `plots.yaml` `map_categories:` — the kind of area (volcanic, regional,
   seismic, monitoring, infrastructure)
5. `plots.yaml` `map_areas:` — the area
6. the map product
7. the run
8. the command line

Layers add up from level to level. `-NAME` takes a layer or a preset out
again.

## 2. Finding what is there

```bash
gpsplot map station --layers list      # layers and presets; [missing] = file not on this host
gpsplot map station --gmt list         # drawing steps, their PyGMT call and options
gpsplot map station --dem-cmap list    # colour palettes
gpsplot run --dry-run                  # every job of the daily run, all options spelled out
gpsplot run --dry-run --product svinafellsheidi_track
gpsplot map --help                     # everything else
```

## 3. Tutorial: a new area map

1. **Draw it by hand first.**
   ```bash
   gpsplot map station SVIE SKFC SLEC --region=-17.06/-16.58/63.905/64.05 \
       --dem iceland --layers volcanic,rural --save png
   ```
   Use `--region=` with `=` (the west longitude is negative).
2. **Try layers and looks.** Add or take out layers:
   `--layers volcanic,rural --layers=-buildings`. Change one layer:
   `--gmt calderas:pen=2p,red`. Change a GMT default for the whole map:
   `--gmt-config MAP_FRAME_TYPE=plain`.
3. **Put it in `plots.yaml`.**
   ```yaml
   products:
     my_map:
       tool: map
       map: station                  # or velocity, deformation
       area: svinafellsheidi         # station_areas.yaml cdn slug: dir + area defaults
       stations: [SVIE, SKFC, SLEC]
       layers: [-buildings]          # change the area's layers
       gmt: {calderas: {pen: 2p,red}}
       args:
         - --region -17.06/-16.58/63.905/64.05
       runs:
         - name: svinafellsheidi-kort  # the file name
   ```
4. **Check it.** `gpsplot run --dry-run --product my_map`, then
   `gpsplot run --product my_map`.
5. **Deploy it.** Copy `plots.yaml` to `~/.config/gpsconfig/`. The map
   shows in aflogun under Kort for that area, with its caption.

**A velocity map** (`map: velocity`) takes windows counted back from
today: `--last 160 --last 91`, or `--last 182:91 --last 91` for "the 3
months before the last 3". `ts_stations: [SENG, ...]` adds a time series per
station with each window's rate drawn on the data (Kort → Tímaraðir
korta).

**A track map** (`map: station` with `--track EVENT`) shows the daily
positions since an event in events.yaml. `--track-ref STA` measures them
against a reference station; `--track-ref A,B,C` against the mean of several
(a station of them that moves is left out, with a message), and
`ts_stations:` adds each station's series. Each station's trend before the
event is removed, so only a change at the onset shows; `--track-trend keep`
removes only the seasonal (on the reference stations too) and writes each
moving station's rate before the event in the legend (grey): creep before
the event shows. The rates' σ include how much the daily noise wanders
(block means of the residuals before the event). A station with fewer
than 5 days since the event gets a black arrow to its newest position (at
the dots' scale) and a dashed 1-σ ellipse; with no days just before the
event its zero is the trend carried across the gap, and that σ is in the
ellipse. The legend gives the newest day (green through yesterday, else
red). `--inset W/E/S/N` adds a
zoom inset with the tracks inside it drawn at its own scale (its DEM
resampled and smoothed); `--inset-corner BL` puts it inside the map on the
legend, `--inset-width` sizes it, `--locator-corner BR` moves the Iceland
map.

**Captions** are made from the map's options (place, dates, horizontal or
vertical). `label:` on the product names the place; `caption:` on a run
replaces the text.

## 4. Tutorial: a new layer

1. **Get the data** as WGS84 GeoJSON (polygons, lines or points), or a text
   file of `LON LAT NAME` lines for names. Put it under `gis_dir`, for
   example `vector/mine/huts.geojson`.
2. **Add it to `map_layers.yaml`:**
   ```yaml
   layers:
     huts:
       file: vector/mine/huts.geojson
       help: mountain huts (FÍ)
       symbol: t0.15c          # points: a GMT symbol
       fill: orange
       pen: 0.3p,black
       gmt: {transparency: 20} # any PyGMT option of its drawing call
   presets:
     hiking: [roads, huts]
   ```
3. **Check it:** `gpsplot map station --layers list` shows it, and
   `--layers huts` draws it. It is also a `--gmt` step: `--gmt huts:t=50`.
4. **If it comes from a web service**, write a `build_NN_*.sh` that fetches
   it, and add the step to `refresh_layers.sh` (`FILES` and `SCRIPT`), so
   the production host keeps it fresh.

## 5. Reference

**`map_layers.yaml`** — `layers:` NAME: `file`, `help`, `pen`, `fill`,
`label` (a feature property to write), `font`, `symbol` (points), `build`
(what makes the file), `gmt` (more PyGMT options). `presets:` NAME: a list
of layers. `none` means no pen or no fill.

**`plots.yaml` map product** — `tool: map`, `map`, `area` or `dir`,
`label`, `stations`, `ts_stations`, `layers`, `gmt`, `gmt_config`,
`args`, `runs`.
**Run** — `name`, `caption`, `publish`, `layers`, `gmt`, `gmt_config`,
`args`.
**`map_categories:` / `map_areas:`** — `layers`, `gmt`, `gmt_config`,
`args`.
**`defaults:`** — `save`, `map_args`, `map_layers`, `map_gmt`,
`map_gmt_config`.

`gmt:` is `{STEP: {KEY: VALUE}}` or a list of `STEP:KEY=VALUE`;
`gmt_config:` is `{KEY: VALUE}` (GMT defaults, `gmt defaults -D` lists them).

**Drawing steps:** `gpsplot map station --gmt list`.

## 6. Where the data comes from

| What | How it gets to a host |
|---|---|
| FutureVolc volcanoes, calderas, swarms | fetched: `build_07_volcanic.sh` |
| Reykjanes lavas, fissures, graben, barriers | fetched: `build_08_reykjanes.sh` |
| IS 50V roads, buildings, farm names | fetched: `build_09_is50v.sh` |
| Iceland DEM + coast, glaciers, lakes | shipped: `gis/dem/fetch_dem.sh` (package `iceland-dem` on git.vedur.is, never GitHub) |

`gis/vector/refresh_layers.sh` re-fetches the first three; the
`gps-gis-refresh` timer runs it monthly (TODO: install on the production
host). A failed or shrunken download keeps the old files.

aflogun's interactive map has its own layers (`gislayers.json`) and the
VÍ GIS list (api.vedur.is/gis/layers, kept an hour).

## 7. When something is missing

- **A layer file is missing:** the map is drawn without it, with a warning.
  `--strict-layers` stops instead.
- **The Iceland DEM is missing:** GMT's own `@earth_relief_15s` is drawn
  (coarser, about 450 m). GMT downloads it once into `~/.gmt`.
- **No IS 50V coast:** GMT's own coast.
- **A web service is down:** the refresh keeps the old files; aflogun keeps
  the last layer list.

## 8. Troubleshooting

- **`--gmt` refused:** the message names the nearest step or option.
  `--gmt list` shows them.
- **"cannot be used with":** an option given twice, once as a long name
  and once as a letter. Use one form.
- **A layer is drawn but hidden:** layers are drawn in order, so a later one
  covers an earlier one. Change the order in `layers:`.
- **Names are missing:** names that would overlap are dropped (the first
  one wins), and names under the logo or cut by the frame are left out.
- **`--layers -x` fails:** write `--layers=-x` (the `=` is needed).

## TODO: inline help in the config files

- `plots.yaml`: a short key list on the map products (the `map_categories`
  block has one).
- `map_layers.yaml`: an example of a new layer and of `gmt:`.
- `postprocess.cfg`: a comment on `gis_dir`.
