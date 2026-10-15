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
(`~/gps-data/dem`, its README; `dem_dir` in postprocess.cfg `[PATHS]`
overrides) at the level the map needs: the coarsest of 1″/3″/15″ that covers
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

How the ice is drawn (`maps._glacier_composite`):
- Glacier cells move up by 100 000 m into a white slice appended to the
  CPT, so land, sea and ice draw in **one** `grdimage`.
- A second, NaN-transparent ice image came out red everywhere outside the
  ice: GMT `-Q` fails on such a grid here.
- GMT also rejects a CPT with a gap, so the white slice starts where the
  land colours end.
- `xarray.where` drops PyGMT's registration and geographic flags, so they
  are restored.

`--dem-cmap slate` is the blue-grey land of the Bárðarbunga paper maps. The
`lightgrey` land is now 200 instead of 225, so white ice stands out.

Any `--dem` grid is drawn with its NaN cells transparent and **no water
painted over it**, so a topo-bathymetry grid shows its sea floor. A
land-only grid still gets the flat water fill. `--dem-cmap` is `lightgrey`
(default: grey land, blue sea floor), `slate`, `relief`, or any GMT CPT. The hillshade
is `-Ne0.6`; the old `-Nt1.5` went near black on steep slopes.

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
