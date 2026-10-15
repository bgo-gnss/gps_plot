# gps_plot — the map lane (PyGMT)

Deep context for `maps.py`, routed out of `gps_plot/CLAUDE.md` when that file
passed its 150-line subdir budget a second time. **Moved verbatim** — the
conventions below are what keep four map functions from each inventing their
own coordinate source and colour vocabulary.

Read `../CLAUDE.md` first for the package summary and cross-references.

---

## Map lane (PyGMT, PLAN Phase 3 — deformation lane live)

`maps.py` — four map functions sharing the slice-1 conventions (coords from
`stations.cfg` via `gps_parser` or pre-resolved `StationCoordinate`s, zero
hardcoding — `DEFAULT_*` params, lazy pygmt guard, returns `pygmt.Figure`,
`outfile=` saves). All take `dem_grid=` for a hillshade background (slice 3).

- `station_map(stations, ...)` — coastline base + station markers/labels.
- `velocity_map(vectors, ...)` — `fig.velo` arrows + 1-σ error ellipses.
  Input: `VelocityVector` records from `velocity_vectors(stations, e, n, σe,
  σn)` (arrays, mm/yr) or `velocity_vectors_from_geojson()` (the `gps_api`
  `GET /velocities` product). WLS formal σ vs MLE honest σ flow through as-is
  (bigger ellipses); mixed `method` tags become color-coded layers
  (`DEFAULT_METHOD_COLORS`) + legend.
- `deformation_vectors(stations, obs_e, obs_n, models, ...)` — observed
  displacement field (mm, NaN = missing) vs N model `VectorField`s, with
  source stars, scale-reference arrow + color-keyed legend. Model fields via
  `mogi_model_field()` / `okada_model_field()` (lazy `gps_analysis` forwards;
  Mogi params match the `gps_api` `DeformationResult` product) or pass
  pre-computed mm arrays. `examples/mogi_vector_comparison.py` (Svartsengi
  observed vs two Mogi models + sample PNG) is now a thin caller of this.
- `slip_map(slip, ...)` — Okada distributed slip as colored fault patches +
  colorbar. Consumes the `gps_api` `SlipDistributionResult` mapping
  (`models/<region>_slip.json`; corners from per-patch lon/lat + plane
  geometry via `slip_patches_from_product()`, metric from
  `gps_analysis.local_coordinates`) or pre-built `SlipPatch` polygons.
  `view="map"` (surface projection) or `view="plane"` (along-strike ×
  down-dip km cross-section — the readable view for near-vertical dikes).

```python
from gps_plot.maps import station_map
station_map(["RHOF", "AKUR"], title="North Iceland", outfile="stations.png")
```

The dormant `gmtplot.py::velomap` recipe is now fully absorbed (velo layers,
DEM hillshade; its `fig.text` lon/lat swap fixed). Render tests are env-gated
on pygmt/GMT (`GMT_LIBRARY_PATH=$HOME/git/gmt/install/lib uv run pytest`).

## Command line: `gpsplot map`

`gpsplot map station|velocity|deformation|slip` (`map.py`) draws these
functions from stations.cfg patterns or a `--region` box: windowed velocity
fits (`--window`, `--last`, `--vertical`; plate removed by default), the stored secular rates, a
gps_api GeoJSON, displacements with Mogi fields, slip products. Design and
measured behaviour: `cli-subcommands-design.md`, step 7.

`--overlay FILE.geojson` (repeatable; `--overlay-pen`, `--overlay-fill`)
draws WGS84 outline layers over the background and under the symbols, e.g.
the LMÍ/NÍ Reykjanes lava flows (WFS `LMI_vektor:goslok_reykjaneselda` on
`gis.natt.is`, listed in the VÍ GIS registry `api.vedur.is/gis/layers`).

`--dem iceland` draws the spliced Iceland topo-bathymetry DEM
from the local GIS directory (`gis_dir` in postprocess.cfg `[PATHS]`,
default `~/gps-data/gis`, its README: `dem/` grids, `vector/` coast,
glaciers and lakes, `cpt/` your colour maps) at the level the map needs: the coarsest of 1″/3″/15″ that covers
the region and is no coarser than one 300-dpi output pixel
(`maps.iceland_dem`). An Iceland overview at 18 cm takes 15″ in ~2 s; the
3″ grid took 20 s and looked the same. It draws the IS 50V coast, not GSHHG:
the full one on local maps, and for overviews one without the islets under
0.2 km², which otherwise blot Breiðafjörður (`build_05_coast.sh`).
Glaciers and lakes follow BGÓ's GMT maps (`gogn/gmt_maps/*/volc-systems.gmt`):
- **Glaciers:** clipped to their outlines, coloured with an all-white CPT
  (`pol.cpt`) and their own gentle two-azimuth hillshade (`-A110/310 -Nt0.4`),
  with a thin outline. On local maps (pixels finer than 5″) they also get
  100 m contours annotated every 300 m.
- **Lakes and wide rivers:** filled as water.

The outlines are IMO's 2025 Sentinel-2 glacier polygons
(`glaciology:glacierpolygon_sent2_2025` on geo.vedur.is, "Útlínur jökla" in
the VÍ GIS registry). The lakes come from IS 50V `vatnafar_flakar`
(`fitjuflokkur` 701 with `yfirbordsvatn` 3 for lakes and 1 for rivers; 4 is
glaciers). IMO's GeoServer has no lake layer. Both are built by
`build_06_ice_water.sh`; `--glaciers` / `--lakes FILE|none` override them.

### Colours: land, sea floor, ice

Three options colour a `--dem` background, each with its own layer:

| Option | Layer | Default |
|---|---|---|
| `--dem-cmap` | land, z ≥ 0 | `lightgrey` |
| `--bathymetry-cmap` | sea floor, z < 0 | the `--dem-cmap`'s sea half when it has one (`geo`, `sealand`, a file reaching below 0), else gpsplot's `sea` blues; `none` = flat water without relief |
| `--glacier-cmap` | ice inside `--glaciers` | `white` |

A name is looked up in this order, first match wins (`palettes.py`):
1. an existing CPT file;
2. `NAME.cpt` in `cpt/` of the GIS directory (yours; it also overrides
   gpsplot's);
3. gpsplot's own `src/gps_plot/cpt/`: `lightgrey`, `slate` (the
   Bárðarbunga maps' blue-grey), `relief`, `sea`;
4. a GMT built-in CPT (`geo`, `sealand`, SCM `batlow`, cmocean `deep`, …);
5. a GMT colour (`white`, `200/200/200`) or colour list (`white,lightblue`).

`--dem-cmap list` prints every name. An unknown name stops before the map is
drawn and suggests close names.

A CPT **file** keeps its own z in metres, GMT's rule for files, and is cut
to the layer's range. To make one for the user directory, run
`gmt makecpt -Cbatlow -T0/2200 > ~/gps-data/gis/cpt/mine.cpt`. A GMT
built-in or a colour is stretched over the layer's range of the map. A
hinged GMT CPT splits at 0: `-T0/max` gives its land half and `-Tmin/0` its
sea half.

How it is drawn (`maps._dem_colours`):
- The three layers are stitched into **one** CPT, so land, sea and ice
  draw in one `grdimage`.
- Glacier cells move up by 100 000 m into the ice slices and take their
  own gentle hillshade.
- A second, NaN-transparent ice image came out red everywhere outside the
  ice: GMT `-Q` fails on such a grid here.
- **GMT rejects a CPT with a gap.** It then aborts `grdimage`, but the
  figure is still saved without its background. So `palettes.stitch`
  writes every boundary from one Python float and fills any gap. Before
  this, re-formatting the land top (2109.0035 as 2109) broke `sealand`
  and `solar`. `palettes.check_gaps` raises before drawing.
- `xarray.where` drops PyGMT's registration and geographic flags, so they
  are restored. It also refuses to align `grdgradient` output with its
  grid (the last digits of the coordinates differ), so the flat-sea
  shading uses numpy.

### Smoother relief: `--dem-smooth KM`

`--dem-smooth KM` runs a Gaussian filter KM wide (`grdfilter -Fg<KM> -D2`)
over the grid before the hillshade is computed. Only the shading is
smoothed: the colours, the coast and the glacier edges come from the
unfiltered grid and stay sharp. 2–4 km suits an Iceland overview, and the
filter costs about a second there. The filter is the `smooth` step, so
`--gmt smooth:filter=m3` gives a median filter instead.

### GMT arguments: `--gmt STEP:KEY=VALUE`, `--gmt-config KEY=VALUE`

Every map is drawn in named steps. `--gmt STEP:KEY=VALUE` gives one GMT
argument to every PyGMT call of that step, or replaces gpsplot's own value.
The flag can be repeated. The steps are coast, smooth, gradient, glacier-gradient,
dem, contours, glaciers, lakes, coastline, overlay, stations, labels,
vectors, legend and logo (`gmt_overrides.LAYERS`).

```bash
gpsplot map station --tos --dem iceland --region -17.9/-16.9/64.45/64.85 \
    --gmt glaciers:pen=1p,blue --gmt contours:W=0.5p,red \
    --gmt labels:F=+f9p,Helvetica-Bold,darkred+jTL --gmt stations:S=c0.3c \
    --gmt-config MAP_FRAME_TYPE=plain --gmt-config FONT_ANNOT_PRIMARY=8p
```

- **KEY** is a PyGMT long name of that call, such as `pen` or `area_thresh`,
  or one GMT letter, such as `W` or `A`; PyGMT's warning about letters is
  silenced. A bare `STEP:KEY` passes a flag. A KEY replaces gpsplot's value
  under either name: the table maps the long names gpsplot passes to their
  letters, and `test_alias_table_pairs_conflict_in_pygmt` proves each pair
  against PyGMT. `text` folds `font` and `justify` into one `-F` without
  complaint, so `labels:F=…` drops both.
- **Checked before drawing:** an unknown step or key stops with suggestions.
  So do the keys gpsplot sets itself: data, `region`/`projection`, and
  `dem:cmap`/`C`, because the glacier slices sit 100 000 m up in the
  stitched CPT. Use `--dem-cmap` and friends for colours.
- **Where a step applies:** `coast` applies to both coast calls, but the
  shoreline-only call over a DEM takes no fill (`land`/`water`) and no
  frame. `stations` applies to every station marker, including all four
  status symbols of a `--tos` map. `vectors` applies to the reference arrow
  of the vector legend too, so it keeps the scale.
- **`--gmt-config`** sets GMT defaults (`pygmt.config`) for the whole map,
  over gpsplot's title font. GMT rejects an unknown key or value when the
  config is entered, before anything is drawn.
- **Cost:** transparency (`dem:t=20`) is slow in GMT, about 40 s instead of
  9 s for a 1″ close-up.
- **Named options:** an option that keeps coming back becomes a named one,
  like `--glacier-cmap`.

Any `--dem` grid is drawn with its NaN cells transparent and **no water
painted over it**, so a topo-bathymetry grid shows its sea floor. A
land-only grid still gets the flat water fill. `lightgrey` is 200, not
225, so white ice stands out. The hillshade is `-Ne0.6`; the old `-Nt1.5`
went near black on steep slopes.

`gpsplot map station --tos` plots the GPS stations of **TOS**, the station
register, instead of stations.cfg. That includes 13 stations stations.cfg
lacks, such as S001, SKA2 and DRA2. Each status gets its own symbol and a
legend with counts: continuous, external, campaign, discontinued
(`sources/tos.py`). The list is cached for a day; `--refresh-tos` refetches
it. TOS attributes are dated, and the current value is the open period
(`date_to` null):
- A `date_end` stored as an open period or as a zero-length one
  (HLFJ, GLER, KALT, STOR) means the station ended.
- A `date_end` period closed later revokes the end (GRVV, GRVM, GJAC).
- When two entities share a marker (SOHO), the running one wins.

Station maps label their stations only up to 60 stations; `--labels` and
`--no-labels` override this.

### Which stations: `--tos-search`, `--stations-file`, `--no-stations`

- **`--tos-search "SEARCH"`** selects TOS stations with a `tos search`
  command line, in one quoted string: every expression and flag of
  `tos search` works (`--epos`, `--active-gps`, `--receiver polarx5`,
  `'name ~ *vík*'`, `contact.owner ~ …`, `--domain`, `--at`). gpsplot hands
  it to `tostools.tos.search_from_argv`, which parses it with the command's
  own parser, so the two never drift apart. The search is limited to
  `subtype = GPS stöð` unless it names a subtype; `'subtype = all'` lifts
  that. A search value may start with `--` (`cli/map.py` joins it to the
  option, as it does for west longitudes).
- **Legend groups** are the status for GPS stations and the subtype for a
  mix (a GPS station keeps the red triangle). `--group-by CODE` groups by
  any open TOS attribute. Groups other than the statuses take the symbols
  of `maps.GROUP_STYLES` in order.
- **`--stations-file FILE[:LABEL]`** (repeatable) reads stations from
  `MARKER LON LAT [NAME]` text, a CSV with marker/lon/lat columns, or
  GeoJSON points (`sources/station_files.py`). Each file is one legend
  group. Alone, the files are the whole map; beside stations, `--tos` or
  `--tos-search`, they are drawn with them, inside `--region`.
- **`--no-stations`** draws no station symbols or names, on station,
  velocity and deformation maps; the stations still set the extent.
  `gpsplot map base` is a station map without stations, Iceland when no
  `--region` is given.

```bash
gpsplot map station --tos-search "--epos" --dem iceland --save pdf
gpsplot map station --tos-search "'subtype = [GPS stöð, SIL stöð]'" --save pdf
gpsplot map station --tos-search "--active-gps" --group-by continuity --save pdf
gpsplot map station --tos --stations-file planned.txt:Planned --save pdf
gpsplot map base --dem iceland --dem-smooth 2 --save pdf
```

```bash
gpsplot map station --tos --dem iceland --region -24.8/-13.2/63.2/66.7 \
    --projection M18c --title "GPS stations in TOS" --save pdf
gpsplot map velocity --region -22.8/-21.9/63.75/64.1 \
    --window 20231027:20231109 --window 20231123:20231205 \
    --dem iceland --save pdf
```

## Installing the lane

The `maps` extra is `pygmt>=0.19` (PyPI) on top of a **system GMT ≥ 6** C
library (OS package, conda-forge `gmt`, or from-source via
`GMT_LIBRARY_PATH`) — `uv sync --extra maps`. Import of `gps_plot` or the
matplotlib lane never needs it: `maps.py` guards the pygmt import lazily, so
a production install without GMT is unaffected. `--tos` needs tostools
(the `tos` extra, or the editable sibling in the gpslibrary env), imported
only when the station list is fetched.

## Cross-References

- `../CLAUDE.md` — package summary, layout, console scripts
- `../../gps_analysis/CLAUDE.md` — Mogi/Okada forward models behind the
  model fields
- `../../gps_api/CLAUDE.md` — the `/velocities` and slip-distribution
  products these consume
- `detrend-lane.md` — the other deep lane (cleaned view, workbench, pickers)
