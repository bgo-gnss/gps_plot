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

### File names: `map_{kind}_{area}_{selection}[_{extras}]`

A map is named after what it shows, not its first station
(`gps_plot.areas`). `--name` still overrides the name.

- **area:** the named box of `areas.AREAS` the map's region overlaps most
  (intersection over union ≥ 0.35): `iceland`, `reykjanes`, `hengill`,
  `katla`, `askja`, `krafla` and others. Otherwise it is the region's own
  coordinates, such as `22.8W-21.9W_63.75N-64.1N`. Add a box to the table
  and maps of that area take its name.
- **selection:** how the stations were chosen, joined by `+`:
  - `tos` for `--tos`;
  - `tos-<search>` for `--tos-search` (`--epos` → `tos-epos`);
  - the station patterns (`SEY?` → `SEY`);
  - the `--stations-file` labels.

  A plain `"*"` adds nothing.
- **extras:** the velocity windows and component, the deformation
  window and `mogi`.

| Map | File |
|---|---|
| `map station --tos-search "--epos" --dem iceland` | `map_station_iceland_tos-epos` |
| `map base` | `map_base_iceland` |
| `map station "*" --region -22.8/-21.9/63.75/64.1` | `map_station_reykjanes` |
| `map velocity --region … --last 7` | `map_velocity_reykjanes_20261009-20261016_horizontal` |

### GMT arguments: `--gmt STEP:KEY=VALUE`, `--gmt-config KEY=VALUE`

Every map is drawn in named steps. `--gmt STEP:KEY=VALUE` gives one GMT
argument to every PyGMT call of that step, or replaces gpsplot's own value.
The flag can be repeated. The steps are coast, smooth, gradient, glacier-gradient,
dem, contours, zones, glaciers, lakes, coastline, overlay, grid, stations, labels,
vectors, legend, frame and logo (`gmt_overrides.LAYERS`).

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
  shoreline-only call over a DEM takes no fill (`land`/`water`). Neither
  coast call draws the frame: `--gmt frame:…` sets the frame and
  `--gmt grid:…` the graticule. `stations` applies to every station marker, including all four
  status symbols of a `--tos` map. `vectors` applies to the reference arrow
  of the vector legend too, so it keeps the scale.
- **`--gmt-config`** sets GMT defaults (`pygmt.config`) for the whole map,
  over gpsplot's title font and frame style. GMT rejects an unknown key or value when the
  config is entered, before anything is drawn.
- **Cost:** transparency (`dem:t=20`) is slow in GMT, about 40 s instead of
  9 s for a 1″ close-up.
- **Named options:** an option that keeps coming back becomes a named one,
  like `--glacier-cmap`.

### Frame, graticule and logo

The frame copies the iceland3 overview map (`gogn/gmt_maps/iceland3/iceland.gmt`
→ `icel-overv.pdf`), scaled from its 30 cm to the 12 cm default
(`maps.DEFAULT_GMT_DEFAULTS`):
- a plain 1.25p black frame;
- ticks and degree-minute annotations inside it, annotated on W and S only
  (`DEFAULT_FRAME = ("WSne", "af")`);
- a dashed grey graticule at the annotations.

The graticule (`grid` step) goes over the base layers, under the stations
and legends. The frame (`frame` step) is drawn last, since anything drawn
after it would hide the annotations inside the map. Both are `basemap` calls.
`+e` skips the annotations at the axis ends, which would pile up in the
corners. The slip plane view, a section rather than a map, keeps its ticks
and annotations outside (`PLANE_GMT_DEFAULTS`). A title rides on the axes
spec (`WSne+t…`): GMT refuses a separate `+t`
beside `WSne` when the title has spaces.

**Oblique maps: `--region LON/LAT/LON/LAT+r`.** The old map's tilted
graticule and upright longitude labels come from its oblique stereographic
projection with a corner region (`-JS-19/65 -R-25/63/-12.8/66.7r`):

```bash
gpsplot map base --dem iceland --dem-cmap icel --bathymetry-cmap lightblue \
    --projection S-19/65/12c --region -25/63/-12.8/66.7+r --save pdf
```

- A `+r` region (or GMT4's trailing `r`) is parsed into the map's
  lower-left and upper-right corners (`map.Corners`). Once the projection
  is known it becomes a `maps.CornerRegion`.
- That object is a tuple: GMT's lon/lat box around the rectangle
  (`mapproject -Wr`). So whatever needs a box takes it unchanged: the DEM
  level, `grdcut`, the scale-bar length.
- The coast call draws from its corner form (`gmt_region`).
- Station and vector filters use `maps.within`. It applies the box, then
  the rectangle itself (`contains`).
- The vector legend is placed by map fraction (`region_point`, `at`).
- `MAP_ANNOT_OBLIQUE=anywhere` writes the longitudes along their meridians,
  as the overview's `OBLIQUE_ANNOTATION = 0` did. East of the central
  meridian they read downwards, as in `icel-overv.pdf`. It has no effect on
  Mercator.
- On an oblique map, the scale bar and the vector legend sit higher, clear
  of the taller upright labels.
- A plain W/E/S/N region with `-JS` keeps GMT's meaning: a meridian and
  parallel boundary, which comes out fan-shaped.

**`--fissure-swarms`: the FutureVolc fissure swarms and central volcanoes**
on any map, as on the iceland3 overview.
- **Fill:** the land inside them is slate blue, from the closed polygons
  (`FISSURE_SWARM_ZONES`). With a DEM the colour sits under the hillshade.
  Without one it is a flat fill, and the sea is painted over it again.
- **Outline:** drawn from the FutureVolc GeoJSON as served
  (`FISSURE_SWARM_OUTLINES`), so no closing edge shows.
- **Building them:** both come from `vector/build_07_volcanic.sh`. If they
  are missing, the map stops with that hint.

```bash
gpsplot map station --tos --fissure-swarms --save pdf
```

**A full-Iceland map is an overview by default.** Neither `--region`
nor `--projection` given, and an extent of at least 8° × 2.5° (`map base`,
`--tos`, every station): the map takes the overview frame.
- **Frame:** `OVERVIEW_PROJECTION` `S-19/65/12c` over the corners
  `OVERVIEW_CORNERS` -25.2/62.9/-12.4/66.9+r, with the longitudes along their
  meridians.
- **Sea:** with a DEM and no `--bathymetry-cmap`, the light-blue sea of
  `OVERVIEW_SEA` under the sea-floor hillshade.
- **When not:** if a station falls outside the frame (Kolbeinsey), the map
  stays a Mercator box (`map._overview`). `--projection M12c` keeps the
  Mercator box too.
- **Logo, legend and scale bar over the sea.** `OVERVIEW_CORNERS`
  (-25.2/62.9/-12.4/66.9) widens the iceland3 frame just enough:
  - the logo, small (`OVERVIEW_LOGO_WIDTH` 1.6 cm), sits bottom left
    between the 24°W and 22°W annotations. The frame is drawn again over
    it, so an annotation stays on top should they meet;
  - the scale bar sits bottom right.
- **DEM by default:** a full-Iceland overview takes `--dem iceland` when
  it is built (`_iceland_dem_built`). `--dem none` keeps flat land.
- **The status legend finds its place** (`gps_plot.layout`): the free
  sea position nearest the top left, else the top right, the bottom left,
  the bottom right. The mask is 1 mm cells of the drawn DEM (z >= 0 is
  land), else of GSHHG land and lakes. The legend keeps clear of:
  - the station symbols;
  - the logo and the scale bar;
  - the degree annotations inside the frame.

  Without a free spot it keeps `jTR`. Mercator maps keep the fixed
  corners.

**Fissure swarms and central volcanoes: `--dem-zone POLYGONS[:CMAP]`.**
The overview recoloured them on the relief itself: `psclip
fisswarms_fil.xy` then `grdimage -Clightslategrey.cpt -Igrad.grd`, then
the outline. `maps.DemZone` does the same in the one DEM pass.
- **Colours:** the land cells inside each zone move up into their own
  slices of the stitched CPT (`ZONE_Z_OFFSET` per zone, above the glacier
  slices) and keep the land hillshade.
- **Overlaps:** sea cells stay sea. A later zone, and the glaciers, take
  the cells they overlap.
- **Outline:** each zone's outline is drawn over the DEM (`zones` step).
  `CMAP` `none` draws the outline alone. Without a DEM, a colour zone is a
  flat fill.
- **Files:** a path that does not exist is looked up under the GIS
  directory. The overview's own polygons are in `gis/vector/iceland3/`,
  copied from `/mnt_data/gps_gmt_data/isl`: 22 swarms, 33 volcanoes,
  `centralvolc2.xy` and the calderas.
- **FutureVolc:** `build_07_volcanic.sh` also writes its swarms and
  volcanoes closed, one polygon per outline, through
  `close_outlines.py`, as `vector/futurevolc_{fissure_swarms,central_volcanoes}_closed.gmt`
  (25 swarms, 28 volcanoes). The ÍSOR 2014 outlines are open and are
  closed explicitly:
  - a U open to the sea, or a nearly closed ring, gets its ends joined;
  - side lines (Krafla, Reykjanes-Svartsengi, Hrómundartindur) become a
    band along the swarm's axis. Where only one of a system's lines runs,
    the band reaches the nearest ÍSOR neighbour line, because a boundary
    two swarms share is drawn once (Hrómundartindur and Hengill).

  A closing edge across the sea is harmless, because a zone recolours only
  land. The raw GeoJSON stays for `--overlay`.

```bash
gpsplot map base --dem iceland --dem-cmap icel --bathymetry-cmap none \
    --projection S-19/65/12c --region -25/63/-12.8/66.7+r \
    --dem-zone vector/iceland3/fisswarms_fil.xy \
    --dem-zone vector/iceland3/centralvolc.xy \
    --dem-zone vector/iceland3/centralvolc2.xy:none --save pdf
```

The logo is the two-line Veðurstofa logo of the RTK plots
(`logos/vi_logos/VI_Two_Line_Blue.png`), 2.5 cm wide at the top left and
30 % transparent, as `rtk_gps` draws it with `alpha=0.7`. Its white ground
gives the faint box.

The old map's land colours are `icel.cpt` in the GIS `cpt/` directory:
- `--dem-cmap icel` gives the relief colours on land and its pale sea half;
- add `--bathymetry-cmap lightblue` for the old flat blue sea;
- `pol.cpt` is its all-white glacier palette (`--glacier-cmap pol`);
- `lightslategrey.cpt` is its blue-grey fill of the fissure swarms and
  central volcanoes, which also works as a land colour.

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
