"""PyGMT deformation maps for the GNSS network (map lane, PLAN Phase 3).

The ``gps_plot`` map lane: PyGMT figures for the deformation products the
aflogun portal and reports consume. Four public map functions share the
same conventions (config-driven coordinates, ``DEFAULT_*`` style
parameters, lazy PyGMT import, returned ``pygmt.Figure``):

* :func:`station_map` — coastline base map + GNSS station markers
  (slice 1; also the base layer of the other maps).
* :func:`velocity_map` — station horizontal-velocity vectors with formal
  1-σ error ellipses (GMT ``velo``), from :class:`VelocityVector` records
  built out of velocity arrays (:func:`velocity_vectors`) or the
  ``gps_api`` ``GET /velocities`` GeoJSON
  (:func:`velocity_vectors_from_geojson`). WLS formal σ and colored-noise
  MLE honest σ flow through unchanged — an MLE ellipse is simply bigger;
  mixed inputs can be color-coded per estimator (``method_colors``).
* :func:`deformation_vectors` — observed displacement field plus one or
  more model vector fields (:class:`VectorField`) overlaid and
  color-coded, with source markers, a scale-reference arrow and a legend.
  Model fields come pre-computed or from the ``gps_analysis`` forward
  models via :func:`mogi_model_field` / :func:`okada_model_field`.
* :func:`slip_map` — an Okada distributed-slip distribution as a colored
  fault-patch map (map view) or fault-plane cross-section
  (``view="plane"``), from the ``gps_api`` ``SlipDistributionResult``
  product (:func:`slip_patches_from_product`) or pre-built
  :class:`SlipPatch` polygons, with a colorbar.

Data provenance
    Station coordinates and display names come from ``stations.cfg`` via
    the :mod:`gps_parser` API (``ConfigParser.getStationInfo``) — the same
    configuration source the rest of the pipeline uses (``receivers``,
    ``gps_api.precompute``). Model math is never re-derived here: forward
    vectors call :func:`gps_analysis.mogi_forward` /
    :func:`gps_analysis.okada_forward`, and the products consumed are the
    ``gps_api`` store artifacts (``velocities/<region>.geojson`` — a
    ``VelocityCollection`` FeatureCollection; ``models/<region>_slip.json``
    — a ``SlipDistributionResult``). Nothing is hardcoded: regions,
    projections, styles, scales and output paths are all parameters
    (module-level ``DEFAULT_*`` constants provide the defaults).

Array conventions at the API boundary
    Positions are geodetic longitude/latitude [decimal degrees, WGS84] in
    ``(N,)`` arrays aligned station-by-station. Velocities are mm/yr,
    displacements mm (NaN = missing observation), slip/opening metres.
    ``gps_analysis`` forward models return ``(3, N)`` metres, rows
    (east, north, up) — the field builders convert to mm horizontals.

Optional dependency
    PyGMT (and the system GMT >= 6 C library underneath it) is an optional
    extra — importing :mod:`gps_plot` or this module works without it;
    only *calling* the map functions requires it. Install with::

        uv sync --extra maps        # or: pip install 'gps_plot[maps]'

    and make sure the GMT shared library is discoverable (system install,
    conda-forge ``gmt``, or ``GMT_LIBRARY_PATH`` for a from-source build).
    :mod:`gps_parser` (marker resolution) and :mod:`gps_analysis` (model
    forwards, slip-patch geometry) are imported lazily too — dev-group
    siblings, needed only by the code paths that use them.

DEM/hillshade background (slice 3)
    All map-view functions take ``dem_grid=`` (a GMT-readable grid file or
    an in-memory grid): the land fill is replaced by a ``grdimage`` of the
    region-cut grid shaded with ``grdgradient`` (azimuth/normalization are
    parameters), with water/shorelines drawn on top. No default grid is
    shipped — the grid path is caller configuration.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gps_plot.gmt_overrides import layer_kw

if TYPE_CHECKING:  # pragma: no cover - typing only, pygmt stays optional
    import pygmt
    from numpy.typing import ArrayLike, NDArray

#: Default map style — parameters of the map functions, never literals in
#: the function bodies. Override per call; change here to restyle globally.
DEFAULT_PROJECTION = "M12c"  # Mercator, 12 cm wide
DEFAULT_RESOLUTION = "auto"  # GSHHG shoreline resolution (auto per region)
DEFAULT_LAND = "lightgrey"
DEFAULT_WATER = "lightblue"
DEFAULT_SHORELINES = "0.2p,black"
#: the frame: annotated W and S, ticked all round (iceland3/iceland.gmt's
#: ``-Ba…WSne``); drawn last, so the annotations inside the map stay on top
DEFAULT_FRAME: tuple[str, ...] = ("WSne", "af")
#: dashed graticule at the annotations (``+n``: their text comes with the
#: frame), over the base layers and under stations and legends
DEFAULT_GRID: tuple[str, ...] = ("ag", "+n")
#: GMT defaults of the iceland3 overview frame (icel-overv.pdf), scaled
#: from its 30 cm to the 12 cm default map: a plain heavy frame, ticks and
#: annotations inside it, degrees and minutes. ``gpsplot map`` sets them
#: under the user's ``--gmt-config``.
DEFAULT_GMT_DEFAULTS: dict[str, str] = {
    "MAP_FRAME_TYPE": "plain",
    "MAP_FRAME_PEN": "1.25p,black",
    "MAP_TICK_LENGTH_PRIMARY": "-0.15c",
    "MAP_TICK_PEN_PRIMARY": "0.5p,black",
    "MAP_ANNOT_OFFSET_PRIMARY": "-0.1c",
    "MAP_GRID_PEN_PRIMARY": "0.25p,gray40,dashed",
    "FORMAT_GEO_MAP": "ddd:mmF",
    "FONT_ANNOT_PRIMARY": "8p,Helvetica,black",
}
DEFAULT_MARKER_STYLE = "d7p"  # diamond, 7 points — gmtplot.velomap heritage
DEFAULT_MARKER_PEN = "1p,red"
DEFAULT_MARKER_FILL = "yellow"
DEFAULT_LABEL_FONT = "5p,Helvetica-Bold"
DEFAULT_LABEL_OFFSET = "J0.2"
#: Padding [degrees] added around the station bounding box when no explicit
#: ``region`` is given.
DEFAULT_MARGIN_DEG = 0.5

#: Vector-layer style (velocity_map / deformation_vectors).
DEFAULT_VELOCITY_SCALE = 0.05  # arrow length [cm on map] per mm/yr
DEFAULT_DISPLACEMENT_SCALE = 0.012  # arrow length [cm on map] per mm
DEFAULT_VELO_CONFIDENCE = 0.39  # GMT -Se confidence: 0.39 = 1-sigma ellipse
DEFAULT_VECTOR_HEAD = "0.4c"  # arrow-head size
DEFAULT_VECTOR_PEN_WIDTH = "1.4p"  # arrow shaft/outline pen width
DEFAULT_VELOCITY_COLOR = "blue"  # fallback vector color (no method mapping)
#: Estimator → arrow color used by :func:`velocity_map` when vectors carry
#: a ``method`` tag (the ``gps_api`` VelocityProperties.method vocabulary).
DEFAULT_METHOD_COLORS: Mapping[str, str] = {
    "wls": "blue",
    "mle": "red",
    "gbis": "purple",
}
DEFAULT_OBSERVED_COLOR = "black"
DEFAULT_VECTOR_STATION_STYLE = "c0.12c"  # small circle under each vector
DEFAULT_VECTOR_STATION_FILL = "white"
DEFAULT_VECTOR_STATION_PEN = "0.6p,black"
DEFAULT_SOURCE_STYLE = "a0.5c"  # star marker for model source positions
DEFAULT_SOURCE_PEN = "0.6p,black"
DEFAULT_LEGEND_FONT = "9p,Helvetica-Bold"
#: Legend anchor as (east-fraction, north-fraction) of the region extent,
#: measured from the south-west corner.
DEFAULT_LEGEND_ANCHOR: tuple[float, float] = (0.03, 0.09)
#: Vertical spacing between legend lines as a fraction of the N-S extent.
DEFAULT_LEGEND_LINE_SPACING = 0.045

#: Slip-map style.
DEFAULT_SLIP_CMAP = "lajolla"  # sequential scientific colour map (GMT builtin)
#: GMT ships lajolla dark->light; reversed puts light at zero slip.
DEFAULT_SLIP_CMAP_REVERSE = True
DEFAULT_PATCH_PEN = "0.25p,gray30"  # fault-patch outline
DEFAULT_PLANE_PROJECTION = "X12c/-6c"  # plane view: x right, depth-axis down
DEFAULT_PLANE_XLABEL = "along-strike distance (km)"
DEFAULT_PLANE_YLABEL = "down-dip distance (km)"

#: DEM/hillshade background (slice 3).
DEFAULT_DEM_CMAP = "geo"  # GMT builtin topo colour map
DEFAULT_DEM_SHADING_AZIMUTH = 315.0  # NW illumination
#: Laplace-normalized hillshade: softer than the velomap heritage -Nt1.5,
#: which went near black on steep slopes (Fagradalsfjall, the fjords)
DEFAULT_DEM_SHADING_NORMALIZE = "e0.6"
#: Glaciers and lakes over the DEM, the way BGÓ's GMT maps drew them
#: (gmt_maps/*/volc-systems.gmt): the ice clipped to its outlines, an
#: all-white CPT and its own gentle two-azimuth hillshade (-A110/310
#: -Nt0.4), a thin outline; lakes as flat water.
DEFAULT_GLACIER_SHADING_AZIMUTH = "110/310"
DEFAULT_GLACIER_SHADING_NORMALIZE = "t0.4"
DEFAULT_GLACIER_FILL = "white"
DEFAULT_GLACIER_PEN = "0.15p,gray55"
#: glacier contours (volc-systems.gmt: grdcontour -C100 -A300 -Wthin,grey)
DEFAULT_GLACIER_CONTOUR_PEN = "thin,grey"
GLACIER_CONTOUR_ANNOTATION_EVERY = 3
DEFAULT_LAKE_FILL = "205/228/242"
DEFAULT_LAKE_PEN = "0.1p,steelblue"
#: glacier heights move by this much into a white CPT segment, so land,
#: sea and ice draw in ONE grdimage (a second, NaN-transparent ice image
#: came out red outside the ice: GMT -Q fails on such a grid here)
GLACIER_Z_OFFSET = 100_000.0

#: pen of a ``coastline`` file (thin and grey: the IS 50V coast has
#: thousands of islets, e.g. in Breiðafjörður)
DEFAULT_COASTLINE_FILE_PEN = "0.15p,gray30"

#: Station symbols per status (``StationCoordinate.status``), in drawing
#: order (bottom first): (style, fill, pen, legend label)
STATUS_STYLES: dict[str, tuple[str, str, str, str]] = {
    "ended": ("t0.16c", "gray60", "0.3p,gray30", "Discontinued"),
    "campaign": ("c0.16c", "white", "0.6p,red", "Campaign"),
    "external": ("t0.22c", "dodgerblue", "0.4p,black", "External"),
    "active": ("t0.22c", "red", "0.4p,black", "Continuous"),
}
DEFAULT_STATUS_LEGEND = "jTR+o0.2c"
#: groups drawn like a status: a GPS station of a mixed TOS map (grouped
#: by subtype) keeps the continuous GPS triangle
GROUP_AS_STATUS = {"GPS stöð": "active"}
#: symbols of the other groups (a TOS subtype, an attribute value, a
#: station file), given in order of first appearance: (style, fill, pen)
GROUP_STYLES: tuple[tuple[str, str, str], ...] = (
    ("s0.2c", "orange", "0.4p,black"),
    ("d0.24c", "purple", "0.4p,black"),
    ("c0.2c", "darkgreen", "0.4p,black"),
    ("i0.22c", "gold", "0.4p,black"),
    ("a0.26c", "deepskyblue", "0.4p,black"),
    ("h0.22c", "brown", "0.4p,black"),
    ("n0.22c", "magenta", "0.4p,black"),
    ("t0.22c", "cyan", "0.4p,black"),
)

#: The local GIS directory (``gis_dir``; ~/gps-data/gis, its README) holds
#: the DEM grids, the vector layers drawn over them and the user's CPTs.
GIS_DEM, GIS_VECTOR, GIS_CPT = "dem", "vector", "cpt"
#: The spliced Iceland topo-bathymetry DEM (gis_dir/dem, its README): one
#: file per level, (file, cell [deg], W/E/S/N coverage), finest first.
ICELAND_DEM_LEVELS: tuple[tuple[str, float, tuple[float, float, float, float]], ...] = (
    ("iceland_topobathy_1s.nc", 1 / 3600, (-25.0, -13.0, 63.0, 66.8)),
    ("iceland_topobathy_3s.nc", 3 / 3600, (-28.0, -10.0, 62.0, 68.0)),
    ("iceland_topobathy_15s.nc", 15 / 3600, (-34.0, -6.0, 60.0, 70.0)),
)
#: the IS 50V coast (WGS84), in full and without the smallest islets
#: (gis_dir/vector, build_05_coast.sh)
ICELAND_COAST = "vector/is50v_coast_wgs84.gmt"
ICELAND_COAST_OVERVIEW = "vector/is50v_coast_wgs84_overview.gmt"
#: IMO glacier polygons (Sentinel-2) and IS 50V lakes + wide rivers, in
#: full and without those under 0.5 km² (build_06_ice_water.sh)
ICELAND_GLACIERS = "vector/imo_glaciers_wgs84.gmt"
ICELAND_LAKES = "vector/is50v_lakes_wgs84.gmt"
ICELAND_LAKES_OVERVIEW = "vector/is50v_lakes_wgs84_overview.gmt"
#: contour interval [m] on the glaciers of local (non-overview) maps
ICELAND_GLACIER_CONTOUR_M = 100.0
#: output pixels coarser than this [deg] take the overview coast
COAST_OVERVIEW_PIXEL_DEG = 5 / 3600
#: the raster resolution a map is planned for [dpi] (PyGMT's PNG default)
DEFAULT_RASTER_DPI = 300
DEFAULT_MAP_WIDTH_CM = 12.0

#: Outline overlays (lava flows, fissures, …) drawn on the base layer.
DEFAULT_OVERLAY_PEN = "0.6p,darkred"
DEFAULT_OVERLAY_FILL = "orange@60"  # 60 % transparent


def _import_pygmt() -> Any:
    """Import pygmt lazily, with an actionable error when unavailable.

    PyGMT raises ``ImportError`` when the Python package is missing but a
    ``GMTCLibNotFoundError`` (not an ``ImportError``) when the package is
    present and the GMT C library is not — both are mapped to one clear
    ``ImportError`` so callers and tests have a single failure mode.
    """
    try:
        import pygmt
    except Exception as exc:  # noqa: BLE001 - see docstring: two failure modes
        raise ImportError(
            "gps_plot.maps requires the optional 'maps' extra "
            "(PyGMT on top of a system GMT >= 6). Install it with "
            "`uv sync --extra maps` (or `pip install 'gps_plot[maps]'`) "
            "and make sure the GMT shared library is discoverable "
            "(e.g. set GMT_LIBRARY_PATH for a from-source GMT build). "
            f"Underlying error: {exc}"
        ) from exc
    return pygmt


def _import_gps_parser() -> Any:
    """Import gps_parser lazily (mirrors ``plot_gps_timeseries``).

    ``gps_parser`` needs a deployed ``~/.config/gpsconfig/`` (or
    ``$GPS_CONFIG_PATH``), so it is imported only when station markers must
    be resolved to coordinates — passing pre-resolved
    :class:`StationCoordinate` objects avoids the import entirely.
    """
    try:
        import gps_parser
    except ImportError as exc:
        raise ImportError(
            "resolving station markers to coordinates requires gps_parser "
            "(https://github.com/bennigo/gps_parser) with a deployed "
            "gpsconfig (stations.cfg); alternatively pass StationCoordinate "
            "objects directly."
        ) from exc
    return gps_parser


def _import_gps_analysis() -> Any:
    """Import gps_analysis lazily (model forwards + local-frame geometry).

    Needed only by the model-field builders (:func:`mogi_model_field`,
    :func:`okada_model_field`) and the slip-patch corner geometry
    (:func:`slip_patches_from_product`) — passing pre-computed vectors or
    :class:`SlipPatch` polygons avoids the import entirely.
    """
    try:
        import gps_analysis
    except ImportError as exc:
        raise ImportError(
            "computing model vectors / fault-patch geometry requires the "
            "gps_analysis leaf library (gpslibrary sibling, dev group: "
            "`uv sync`); alternatively pass pre-computed VectorField / "
            "SlipPatch objects."
        ) from exc
    return gps_analysis


@dataclasses.dataclass(frozen=True)
class StationCoordinate:
    """One station's map position from ``stations.cfg``.

    Attributes:
        marker: 4-character station marker, e.g. ``"RHOF"``.
        lon: Geodetic longitude [decimal degrees, east positive].
        lat: Geodetic latitude [decimal degrees, north positive].
        name: Human-readable station name (``station_name``), or None.
        status: Map group: a status of :data:`STATUS_STYLES` (from TOS
            via :mod:`gps_plot.sources.tos`), any other label (a TOS
            subtype, a station file's label; drawn in the next of
            :data:`GROUP_STYLES`), or None. A map whose stations carry a
            group draws one symbol per group, with a legend.
    """

    marker: str
    lon: float
    lat: float
    name: str | None = None
    status: str | None = None


def station_coordinates(
    markers: Sequence[str],
    *,
    parser: Any | None = None,
) -> tuple[StationCoordinate, ...]:
    """Resolve station markers to coordinates via :mod:`gps_parser`.

    Reads ``latitude`` / ``longitude`` / ``station_name`` from each
    station's ``stations.cfg`` section (``ConfigParser.getStationInfo``),
    the same access path ``gps_api.precompute`` uses — inline-comment
    stripping and duplicate-key handling stay in one place.

    Args:
        markers: Station markers to resolve (e.g. ``["RHOF", "AKUR"]``).
        parser: An existing ``gps_parser.ConfigParser`` to reuse; a new one
            is constructed when None (requires ``$GPS_CONFIG_PATH`` or
            ``~/.config/gpsconfig/``).

    Returns:
        One :class:`StationCoordinate` per marker, input order preserved.

    Raises:
        KeyError: When a marker has no section in ``stations.cfg``.
        ValueError: When a station section carries no usable
            latitude/longitude.
    """
    if parser is None:
        parser = _import_gps_parser().ConfigParser()

    known = {str(s) for s in parser.getStationInfo()}
    coords: list[StationCoordinate] = []
    for marker in markers:
        if marker not in known:
            raise KeyError(
                f"station {marker!r} has no section in stations.cfg "
                f"({parser.get_stations_config_path()})"
            )
        # getStationInfo(marker) wraps the section as {"station": {...}}.
        info = dict(dict(parser.getStationInfo(marker))["station"])
        try:
            lat = float(info["latitude"])
            lon = float(info["longitude"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"station {marker!r}: stations.cfg carries no usable "
                "latitude/longitude — cannot place it on the map"
            ) from None
        name = info.get("station_name")
        coords.append(
            StationCoordinate(
                marker=marker,
                lon=lon,
                lat=lat,
                name=str(name) if name else None,
            )
        )
    return tuple(coords)


def _resolve_stations(
    stations: Sequence[str] | Sequence[StationCoordinate],
    parser: Any | None,
) -> tuple[StationCoordinate, ...]:
    """Markers or pre-resolved coordinates → ``StationCoordinate`` tuple."""
    if not stations:
        raise ValueError("need at least one station")
    if isinstance(stations[0], StationCoordinate):
        return tuple(stations)  # type: ignore[arg-type]
    return station_coordinates([str(s) for s in stations], parser=parser)


def region_around(
    coordinates: Sequence[StationCoordinate],
    *,
    margin: float = DEFAULT_MARGIN_DEG,
) -> tuple[float, float, float, float]:
    """Region ``(west, east, south, north)`` bounding the stations.

    Args:
        coordinates: Stations the region must contain (at least one).
        margin: Padding [degrees] added on every side.

    Returns:
        GMT-style region tuple ``(W, E, S, N)`` in decimal degrees.
    """
    if not coordinates:
        raise ValueError("cannot derive a region from zero stations")
    lons = [c.lon for c in coordinates]
    lats = [c.lat for c in coordinates]
    return (
        min(lons) - margin,
        max(lons) + margin,
        min(lats) - margin,
        max(lats) + margin,
    )


def _region_of_points(
    lons: Sequence[float], lats: Sequence[float], margin: float
) -> tuple[float, float, float, float]:
    """Region ``(W, E, S, N)`` bounding arbitrary points + margin [deg]."""
    if len(lons) == 0:
        raise ValueError("cannot derive a region from zero points")
    return (
        float(min(lons)) - margin,
        float(max(lons)) + margin,
        float(min(lats)) - margin,
        float(max(lats)) + margin,
    )


#: a -B axes spec: which sides to draw (``WSne``), with its modifiers
_SIDES = re.compile(r"^[WESNZwesnzlrbtu]+(\+|$)")


def _frame_args(frame: Sequence[str], title: str | None) -> list[str]:
    """Base-map frame spec list, with the title appended as ``+t`` — to
    the axes spec (``WSne+t…``) when there is one: GMT refuses a separate
    ``+t`` element beside it."""
    args = list(frame)
    if title is not None:
        sides = next((k for k, a in enumerate(args) if _SIDES.match(a)), None)
        if sides is None:
            args.append(f"+t{title}")
        else:
            args[sides] += f"+t{title}"
    return args


def _draw_frame(fig: Any, frame_args: Sequence[str]) -> None:
    """The map frame, drawn last: its annotations sit inside the map
    (:data:`DEFAULT_GMT_DEFAULTS`), so anything drawn after would hide them."""
    fig.basemap(**layer_kw("frame", frame=list(frame_args)))


@dataclasses.dataclass(frozen=True)
class MapOverlay:
    """A GeoJSON outline layer (WGS84 lon/lat) drawn on the base layer.

    Polygons are filled with ``fill`` (None: outline only) and outlined
    with ``pen``; lines get the pen; points are ignored.
    """

    path: str
    pen: str = DEFAULT_OVERLAY_PEN
    fill: str | None = DEFAULT_OVERLAY_FILL


def overlay_segments(geojson: Mapping[str, Any]) -> tuple[str, str]:
    """``(polygons, lines)`` of a GeoJSON as GMT multisegment text.

    Polygon holes get a ``-Ph`` segment header, so GMT leaves them
    unfilled. Raises ValueError for a declared non-WGS84 CRS.
    """
    crs = str(((geojson.get("crs") or {}).get("properties") or {}).get("name", ""))
    if crs and not re.search(r"CRS84|EPSG:?:?4326", crs):
        raise ValueError(f"overlay must be WGS84 lon/lat, not {crs}")
    if geojson.get("type") == "FeatureCollection":
        geoms = [f.get("geometry") for f in geojson.get("features") or ()]
    elif geojson.get("type") == "Feature":
        geoms = [geojson.get("geometry")]
    else:
        geoms = [geojson]
    polys: list[str] = []
    lines: list[str] = []

    def ring(out: list[str], header: str, coords: Sequence[Sequence[float]]) -> None:
        out.append(header)
        out.extend(f"{c[0]:.7f} {c[1]:.7f}" for c in coords)

    stack = [g for g in geoms if g]
    while stack:
        g = stack.pop(0)
        kind, coords = g.get("type"), g.get("coordinates")
        if kind == "GeometryCollection":
            stack.extend(g.get("geometries") or ())
        elif kind in ("Polygon", "MultiPolygon"):
            for rings in [coords] if kind == "Polygon" else coords:
                for k, r in enumerate(rings):
                    ring(polys, ">" if k == 0 else "> -Ph", r)
        elif kind in ("LineString", "MultiLineString"):
            for r in [coords] if kind == "LineString" else coords:
                ring(lines, ">", r)
    return "\n".join(polys), "\n".join(lines)


def _draw_overlays(fig: Any, overlays: Sequence[MapOverlay]) -> None:
    import json
    import tempfile

    for ov in overlays:
        with open(ov.path, encoding="utf-8") as fh:
            polys, lines = overlay_segments(json.load(fh))
        with tempfile.TemporaryDirectory() as tmp:
            for text, closed in ((polys, True), (lines, False)):
                if not text:
                    continue
                seg = os.path.join(tmp, "segments.txt")
                with open(seg, "w", encoding="utf-8") as fh:
                    fh.write(text + "\n")
                kw: dict[str, Any] = {"pen": ov.pen}
                if closed and ov.fill:
                    kw["fill"] = ov.fill
                fig.plot(**layer_kw("overlay", data=seg, **kw))


def map_width_cm(projection: str) -> float | None:
    """Map width [cm] of a GMT projection given by width (``M12c``,
    ``M-19/65/18c``, ``X12c/-6c``); None for a scale form (``m1:…``)."""
    if not projection or projection[0].islower():
        return None
    tokens = projection[1:].split("/")
    token = tokens[0] if projection[0] == "X" else tokens[-1]
    m = re.fullmatch(r"([\d.]+)([cip]?)", token)
    if m is None:
        return None
    value = float(m.group(1))
    return {"": value, "c": value, "i": value * 2.54, "p": value / 72 * 2.54}[
        m.group(2)
    ]


def output_pixel_deg(
    region: Sequence[float], projection: str, dpi: int = DEFAULT_RASTER_DPI
) -> float:
    """Longitude span of one output pixel [deg]."""
    width = map_width_cm(projection) or DEFAULT_MAP_WIDTH_CM
    w, e = region[0], region[1]
    return (e - w) / (width / 2.54 * dpi)


@dataclasses.dataclass(frozen=True)
class DemChoice:
    """A DEM level and the vector layers to draw with it (None: absent)."""

    grid: str
    coastline: str | None
    glaciers: str | None = None
    lakes: str | None = None
    glacier_contours: float | None = None


def iceland_dem(
    gis_dir: str | os.PathLike[str],
    region: Sequence[float],
    projection: str = DEFAULT_PROJECTION,
    *,
    dpi: int = DEFAULT_RASTER_DPI,
) -> DemChoice:
    """The Iceland DEM level for a map, from the GIS directory ``gis_dir``:
    no finer than the output pixels.

    Picks the coarsest level of :data:`ICELAND_DEM_LEVELS` that covers
    ``region`` and whose cell is no larger than one output pixel (an
    Iceland overview at 18 cm takes the 15″ grid in ~2 s; the 3″ grid took
    20 s and looked the same). With the IS 50V coast, the IMO glaciers and
    the IS 50V lakes when present: the overview coast and lakes for pixels
    coarser than :data:`COAST_OVERVIEW_PIXEL_DEG`, glacier contours
    (:data:`ICELAND_GLACIER_CONTOUR_M`) only on finer ones.

    Raises:
        FileNotFoundError: the chosen level is not built.
    """
    base = Path(gis_dir).expanduser()
    w, e, s, n = region
    pixel = output_pixel_deg(region, projection, dpi)
    covering = [
        lv for lv in ICELAND_DEM_LEVELS
        if lv[2][0] <= w and e <= lv[2][1] and lv[2][2] <= s and n <= lv[2][3]
    ] or [ICELAND_DEM_LEVELS[-1]]  # fmt: skip
    fitting = [lv for lv in covering if lv[1] <= pixel]
    name = (fitting[-1] if fitting else covering[0])[0]
    grid = base / GIS_DEM / name
    if not grid.is_file():
        raise FileNotFoundError(
            f"{grid} is not built (see build_04_levels.sh in {grid.parent})"
        )
    overview = pixel > COAST_OVERVIEW_PIXEL_DEG

    def layer(full: str, small: str | None = None) -> str | None:
        path = base / (small if overview and small else full)
        return str(path) if path.is_file() else None

    glaciers = layer(ICELAND_GLACIERS)
    return DemChoice(
        str(grid),
        layer(ICELAND_COAST, ICELAND_COAST_OVERVIEW),
        glaciers,
        layer(ICELAND_LAKES, ICELAND_LAKES_OVERVIEW),
        None if overview or glaciers is None else ICELAND_GLACIER_CONTOUR_M,
    )


def _draw_basemap(
    pygmt_mod: Any,
    fig: Any,
    *,
    region: tuple[float, float, float, float],
    projection: str,
    resolution: str,
    land: str,
    water: str,
    shorelines: str,
    dem_grid: Any | None = None,
    dem_cmap: str = DEFAULT_DEM_CMAP,
    bathymetry_cmap: str | None = None,
    glacier_cmap: str = DEFAULT_GLACIER_FILL,
    dem_shading_azimuth: float = DEFAULT_DEM_SHADING_AZIMUTH,
    dem_shading_normalize: str = DEFAULT_DEM_SHADING_NORMALIZE,
    dem_smooth: float = 0.0,
    overlays: Sequence[MapOverlay] = (),
    coastline: str | None = None,
    glaciers: str | None = None,
    lakes: str | None = None,
    glacier_contours: float | None = None,
) -> None:
    """Draw the shared coast/DEM base layer (slice-3 seam, now live).

    Without ``dem_grid`` this is the slice-1 coastline base (flat land
    fill). With ``dem_grid`` (a GMT-readable grid file path or an
    in-memory grid) a hillshaded ``grdimage`` goes over that base: the
    grid is cut to the region, shaded with ``grdgradient``
    (``dem_shading_azimuth`` illumination, ``dem_shading_normalize``
    amplitude normalization; from the grid smoothed ``dem_smooth`` km
    wide when set, so the relief softens while coasts and colours stay
    sharp) and coloured by :func:`_dem_colours`: land
    ``dem_cmap``, sea floor ``bathymetry_cmap``, ice ``glacier_cmap``.
    Its NaN cells are transparent, so a land-only grid keeps the flat
    water fill while a topo-bathymetry grid shows its sea floor — water
    is NOT painted over the grid. The cells inside ``glaciers``
    (polygons) take ``glacier_cmap`` and their own gentle hillshade;
    without a DEM the glacier polygons are filled with that colour. ``lakes`` are filled as
    water. The shoreline goes on top: ``coastline`` (a line/polygon file,
    e.g. the IS 50V coast) or GSHHG.
    """
    # one coast call when GSHHG draws the shoreline straight over the fill
    plain = dem_grid is None and coastline is None
    fig.coast(
        **layer_kw(
            "coast",
            region=region,
            projection=projection,
            resolution=resolution,
            land=land,
            water=water,
            **({"shorelines": shorelines} if plain else {}),
        )
    )
    if dem_grid is not None:
        subgrid = pygmt_mod.grdcut(dem_grid, region=region)
        relief = subgrid
        if dem_smooth > 0:
            # km on a lon/lat grid, the x spacing scaled by cos(lat)
            relief = pygmt_mod.grdfilter(
                **layer_kw(
                    "smooth", grid=subgrid, filter=f"g{dem_smooth:g}", distance=2
                )
            )
        gradient = pygmt_mod.grdgradient(
            **layer_kw(
                "gradient",
                grid=relief,
                azimuth=dem_shading_azimuth,
                normalize=dem_shading_normalize,
            )
        )
        subgrid, gradient, cpt, ice = _dem_colours(
            pygmt_mod,
            subgrid,
            gradient,
            land=dem_cmap,
            sea=bathymetry_cmap,
            glacier=glacier_cmap,
            glaciers=glaciers,
            water=water,
        )
        fig.grdimage(
            **layer_kw(
                "dem", grid=subgrid, shading=gradient, cmap=cpt, nan_transparent=True
            )
        )
        if glacier_contours and ice is not None:
            fig.grdcontour(
                **layer_kw(
                    "contours",
                    grid=ice,
                    levels=glacier_contours,
                    annotation=glacier_contours * GLACIER_CONTOUR_ANNOTATION_EVERY,
                    pen=DEFAULT_GLACIER_CONTOUR_PEN,
                )
            )
    if glaciers is not None:
        fill = None
        if dem_grid is None:  # a flat fill: a colour, not a CPT
            from gps_plot import palettes

            is_colour = palettes.resolve(glacier_cmap).kind == palettes.COLOUR
            fill = glacier_cmap if is_colour else DEFAULT_GLACIER_FILL
        fig.plot(
            **layer_kw("glaciers", data=glaciers, pen=DEFAULT_GLACIER_PEN, fill=fill)
        )
    if lakes is not None:
        fig.plot(
            **layer_kw(
                "lakes", data=lakes, fill=DEFAULT_LAKE_FILL, pen=DEFAULT_LAKE_PEN
            )
        )
    if coastline is not None:
        fig.plot(
            **layer_kw("coastline", data=coastline, pen=DEFAULT_COASTLINE_FILE_PEN)
        )
    elif not plain:
        # the shoreline over the DEM: no land/water fill, no second frame
        fig.coast(
            **layer_kw(
                "coast",
                exclude=("land", "G", "water", "S", "frame", "B"),
                shorelines=shorelines,
                resolution=resolution,
            )
        )
    _draw_overlays(fig, overlays)
    fig.basemap(**layer_kw("grid", frame=list(DEFAULT_GRID)))


def _dem_colours(
    pygmt_mod: Any,
    grid: Any,
    gradient: Any,
    *,
    land: str,
    sea: str | None = None,
    glacier: str = DEFAULT_GLACIER_FILL,
    glaciers: str | None = None,
    water: str = DEFAULT_WATER,
) -> tuple[Any, Any, str, Any | None]:
    """``(grid, shading, cpt, ice)``: the DEM coloured in one grdimage pass.

    One CPT carries three layers (:mod:`gps_plot.palettes`): ``sea`` below
    0, ``land`` above, and ``glacier`` on the cells inside ``glaciers``,
    which move up by :data:`GLACIER_Z_OFFSET` into their own slices and
    take a gentle two-azimuth hillshade
    (:data:`DEFAULT_GLACIER_SHADING_AZIMUTH`) instead of the land one —
    the clip + white CPT + -Nt0.4 recipe of BGÓ's GMT maps.

    ``sea`` None: the land map's own sea half when it has one (geo,
    sealand, a file reaching below 0), else :data:`palettes.DEFAULT_SEA`.
    ``sea`` "none": the sea floor flat in ``water``, without relief.

    ``ice`` is the grid's heights on the glaciers (NaN elsewhere), for
    contours; None when no glacier is in the grid.

    Raises:
        ValueError: the stitched CPT has a gap (never expected).
    """
    from gps_plot import palettes

    land_pal = palettes.resolve(land)
    if sea is not None and sea.lower() == palettes.NONE:
        sea_pal = palettes.Palette(sea, palettes.COLOUR, water)
        # (numpy: grdgradient's coordinates differ from the grid's in
        # the last digits, so an xarray where() refuses to align them)
        gradient = gradient.copy(data=np.where(grid.values >= 0, gradient.values, 0.0))
    elif sea is not None:
        sea_pal = palettes.resolve(sea)
    elif palettes.is_hinged(land_pal):
        sea_pal = land_pal
    else:
        sea_pal = palettes.resolve(palettes.DEFAULT_SEA)

    ice = _ice_mask(pygmt_mod, grid, glaciers) if glaciers is not None else None
    ground = grid if ice is None else grid.where(~ice)
    lo, hi = float(ground.min()), float(ground.max())
    parts = []
    if lo < 0:
        parts.append(palettes.segment(pygmt_mod, sea_pal, lo, min(hi, 0.0)))
    if hi > 0 or not parts:
        bottom = max(lo, 0.0)
        parts.append(
            palettes.segment(pygmt_mod, land_pal, bottom, max(hi, bottom + 1.0))
        )
    ice_heights = None
    if ice is not None:
        ice_gradient = pygmt_mod.grdgradient(
            **layer_kw(
                "glacier-gradient",
                grid=grid,
                azimuth=DEFAULT_GLACIER_SHADING_AZIMUTH,
                normalize=DEFAULT_GLACIER_SHADING_NORMALIZE,
            )
        )
        ice_heights = _like_grid(grid.where(ice), grid)
        ice_lo, ice_hi = float(ice_heights.min()), float(ice_heights.max())
        glacier_pal = palettes.resolve(glacier)
        parts.append(
            palettes.shift(
                palettes.segment(
                    pygmt_mod, glacier_pal, ice_lo, max(ice_hi, ice_lo + 1.0)
                ),
                GLACIER_Z_OFFSET,
            )
        )
        grid = _like_grid(grid.where(~ice, grid + GLACIER_Z_OFFSET), grid)
        gradient = _like_grid(gradient.where(~ice, ice_gradient), grid)
    text = palettes.stitch(parts)
    palettes.check_gaps(text)
    return grid, gradient, _cached_cpt(text), ice_heights


def _like_grid(arr: Any, grid: Any) -> Any:
    """``arr`` with ``grid``'s GMT registration/geographic flags (xarray
    ``where`` drops them)."""
    arr.gmt.registration, arr.gmt.gtype = grid.gmt.registration, grid.gmt.gtype
    return arr


def _ice_mask(pygmt_mod: Any, grid: Any, glaciers: str) -> Any | None:
    """Boolean array of ``grid``'s cells inside the ``glaciers`` polygons
    (outline cells count), or None when no cell is."""
    xs = grid.coords[grid.dims[1]].values
    ys = grid.coords[grid.dims[0]].values
    # a gridline mask on the grid's cell centres has its shape for either
    # registration (grdmask has no registration option in PyGMT)
    mask = pygmt_mod.grdmask(
        data=glaciers,
        region=[float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())],
        spacing=[abs(float(xs[1] - xs[0])), abs(float(ys[1] - ys[0]))],
        outside=0,
        edge=1,
        inside=1,
    )
    ice = mask.values.astype(bool)
    mys = mask.coords[mask.dims[0]].values
    if (mys[0] < mys[-1]) != (ys[0] < ys[-1]):
        ice = ice[::-1]
    return ice if ice.any() else None


def _cached_cpt(text: str) -> str:
    """``text`` as a CPT file in the cache, named by its content."""
    import hashlib

    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    digest = hashlib.sha1(text.encode()).hexdigest()[:12]
    path = Path(base) / "gps_plot" / f"glacier_{digest}.cpt"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return str(path)


def _velo_style(
    color: str, scale: float, confidence: float, head: str, pen_width: str
) -> dict[str, str]:
    """GMT ``velo`` kwargs for one vector layer (shared arrow styling)."""
    return {
        "spec": f"e{scale}/{confidence}/0",
        "pen": f"{pen_width},{color}",
        "vector": f"{head}+p{pen_width},{color}+e+g{color}",
    }


def _draw_vector_layer(
    fig: Any,
    lon: NDArray[np.float64],
    lat: NDArray[np.float64],
    east: NDArray[np.float64],
    north: NDArray[np.float64],
    *,
    sigma_east: NDArray[np.float64] | None = None,
    sigma_north: NDArray[np.float64] | None = None,
    correlation: NDArray[np.float64] | None = None,
    color: str,
    scale: float,
    confidence: float,
    head: str,
    pen_width: str,
    ellipse_fill: str | None = None,
    ellipse_pen: str | None = None,
) -> None:
    """One ``fig.velo`` layer; rows with non-finite components are dropped.

    The ellipses are outlined with the arrow pen, or ``ellipse_pen``.
    """
    zeros = np.zeros(lon.shape, dtype=np.float64)
    se = zeros if sigma_east is None else sigma_east
    sn = zeros if sigma_north is None else sigma_north
    corr = zeros if correlation is None else correlation
    finite = np.isfinite(east) & np.isfinite(north)
    if not np.any(finite):
        return
    data = np.column_stack(
        [
            lon[finite],
            lat[finite],
            east[finite],
            north[finite],
            se[finite],
            sn[finite],
            corr[finite],
        ]
    )
    kwargs: dict[str, Any] = _velo_style(color, scale, confidence, head, pen_width)
    if sigma_east is not None or sigma_north is not None:
        # outline the uncertainty ellipse with the arrow pen or its own
        kwargs["line"] = True if ellipse_pen is None else ellipse_pen
        if ellipse_fill is not None:
            kwargs["uncertainty_fill"] = ellipse_fill
    fig.velo(**layer_kw("vectors", data=data, **kwargs))


def _draw_vector_legend(
    fig: Any,
    *,
    region: tuple[float, float, float, float],
    entries: Sequence[tuple[str, str]],
    scale_ref: float | None,
    scale_unit: str,
    scale: float,
    confidence: float,
    head: str,
    pen_width: str,
    font: str,
    anchor: tuple[float, float],
    line_spacing: float,
) -> None:
    """Scale-reference arrow + color-coded legend lines (SW corner block).

    ``entries`` are ``(label, color)`` pairs drawn bottom-up from the
    ``anchor`` (region-fraction coordinates); the reference arrow of
    length ``scale_ref`` (data units) sits above them with its magnitude
    labelled. Positions are region fractions — no hardcoded degrees.
    """
    west, east, south, north = region
    x0 = west + anchor[0] * (east - west)
    y0 = south + anchor[1] * (north - south)
    dy = line_spacing * (north - south)
    for i, (label, color) in enumerate(entries):
        fig.text(
            x=x0,
            y=y0 + i * dy,
            text=label,
            font=f"{font},{color}",
            justify="LM",
        )
    if scale_ref is not None:
        y_arrow = y0 + len(entries) * dy
        _draw_vector_layer(
            fig,
            np.array([x0]),
            np.array([y_arrow]),
            np.array([float(scale_ref)]),
            np.array([0.0]),
            color="black",
            scale=scale,
            confidence=confidence,
            head=head,
            pen_width=pen_width,
        )
        fig.text(
            x=x0,
            y=y_arrow + 0.8 * dy,
            text=f"{scale_ref:g} {scale_unit}",
            font=f"{font},black",
            justify="LM",
        )


def _draw_status_markers(
    fig: Any,
    coordinates: Sequence[StationCoordinate],
    *,
    legend_position: str | None = DEFAULT_STATUS_LEGEND,
) -> None:
    """One symbol per group, counted in a legend: the statuses of
    :data:`STATUS_STYLES` first (bottom), then the other groups in order
    of appearance, each in the next of :data:`GROUP_STYLES`.

    A station without a group is drawn as ``active``.
    """
    groups: dict[str, list[StationCoordinate]] = {k: [] for k in STATUS_STYLES}
    for c in coordinates:
        groups.setdefault(c.status or "active", []).append(c)
    others = [k for k in groups if k not in STATUS_STYLES and k not in GROUP_AS_STATUS]
    for key, members in groups.items():
        if not members:
            continue
        if key in STATUS_STYLES:
            style, fill, pen, label = STATUS_STYLES[key]
        elif key in GROUP_AS_STATUS:
            style, fill, pen, _ = STATUS_STYLES[GROUP_AS_STATUS[key]]
            label = key
        else:
            style, fill, pen = GROUP_STYLES[others.index(key) % len(GROUP_STYLES)]
            label = key
        fig.plot(
            **layer_kw(
                "stations",
                x=[c.lon for c in members],
                y=[c.lat for c in members],
                style=style,
                fill=fill,
                pen=pen,
                label=f"{label} ({len(members)})",
            )
        )
    if legend_position:
        fig.legend(
            **layer_kw("legend", position=legend_position, box="+gwhite@20+p0.4p")
        )


def station_map(
    stations: Sequence[str] | Sequence[StationCoordinate],
    *,
    region: tuple[float, float, float, float] | None = None,
    margin: float = DEFAULT_MARGIN_DEG,
    projection: str = DEFAULT_PROJECTION,
    resolution: str = DEFAULT_RESOLUTION,
    land: str = DEFAULT_LAND,
    water: str = DEFAULT_WATER,
    shorelines: str = DEFAULT_SHORELINES,
    frame: Sequence[str] = DEFAULT_FRAME,
    title: str | None = None,
    marker_style: str = DEFAULT_MARKER_STYLE,
    marker_pen: str = DEFAULT_MARKER_PEN,
    marker_fill: str = DEFAULT_MARKER_FILL,
    labels: bool = True,
    markers: bool = True,
    label_font: str = DEFAULT_LABEL_FONT,
    label_offset: str = DEFAULT_LABEL_OFFSET,
    status_legend: str | None = DEFAULT_STATUS_LEGEND,
    dem_grid: Any | None = None,
    dem_cmap: str = DEFAULT_DEM_CMAP,
    bathymetry_cmap: str | None = None,
    glacier_cmap: str = DEFAULT_GLACIER_FILL,
    dem_shading_azimuth: float = DEFAULT_DEM_SHADING_AZIMUTH,
    dem_shading_normalize: str = DEFAULT_DEM_SHADING_NORMALIZE,
    dem_smooth: float = 0.0,
    overlays: Sequence[MapOverlay] = (),
    coastline: str | None = None,
    glaciers: str | None = None,
    lakes: str | None = None,
    glacier_contours: float | None = None,
    outfile: str | os.PathLike[str] | None = None,
    show: bool = False,
    parser: Any | None = None,
) -> pygmt.Figure:
    """Render a coastline map with GNSS station markers (PyGMT).

    Draws a GSHHG coastline base map (or a hillshaded DEM when
    ``dem_grid`` is given) in the given projection, then one marker per
    station and, optionally, the station marker text next to it. This is
    the map-lane base layer: :func:`velocity_map` and
    :func:`deformation_vectors` draw their vector layers on the same base.

    Args:
        stations: Either station markers (strings — resolved to
            coordinates through :func:`station_coordinates` /
            :mod:`gps_parser`) or pre-resolved :class:`StationCoordinate`
            objects (no config access needed).
        region: Map bounds ``(west, east, south, north)`` [deg]; derived
            from the station bounding box + ``margin`` when None.
        margin: Padding [deg] used when ``region`` is None.
        projection: GMT projection spec (default Mercator ``M12c``).
        resolution: GSHHG shoreline resolution passed to ``Figure.coast``
            (``"auto"``, ``"full"``, ..., or the single-letter forms).
        land: Land fill color (ignored when ``dem_grid`` is given).
        water: Water fill color.
        shorelines: Shoreline pen.
        frame: Base-map frame spec; ``title`` is appended as ``+t`` when
            given.
        title: Optional map title.
        marker_style: GMT symbol spec for the station markers.
        marker_pen: Marker outline pen.
        marker_fill: Marker fill color.
        labels: Draw the station marker next to each symbol.
        markers: Draw the station symbols (False: none, nor labels).
        label_font: Label font spec.
        label_offset: Label offset spec (GMT ``-D`` justify form).
        status_legend: Legend position (GMT ``-D``) when the stations
            carry a status; None = no legend.
        dem_grid: Optional GMT-readable elevation grid (file path or
            in-memory grid) for a hillshade background (see
            :func:`_draw_basemap`).
        dem_cmap: Land colours of the DEM (z >= 0): a CPT file, a
            name in the user's or gpsplot's CPT directory, a GMT CPT or a
            colour (:mod:`gps_plot.palettes`).
        bathymetry_cmap: Sea-floor colours (z < 0), named the same way;
            None = ``dem_cmap``'s sea half when it has one, else gpsplot's
            blues; ``none`` = flat ``water``, no relief.
        glacier_cmap: Glacier colours, named the same way (default white).
        dem_shading_azimuth: Hillshade illumination azimuth [deg].
        dem_shading_normalize: ``grdgradient`` normalization spec.
        dem_smooth: Smooth the hillshade (not the colours) with a
            Gaussian filter this wide [km]; 0 = off.
        overlays: GeoJSON outline layers (:class:`MapOverlay`), drawn
            over the coast/DEM and under the symbols.
        coastline: A GMT/OGR line or polygon file drawn as the shoreline
            instead of GSHHG (e.g. the IS 50V coast of the Iceland DEM).
        glaciers: A glacier polygon file: with a DEM the ice is drawn
            white with its own gentle hillshade (see :func:`_draw_basemap`).
        lakes: A lake polygon file, filled as water.
        glacier_contours: Contour interval [m] drawn on the glaciers
            (annotated every 3rd), or None.
        outfile: Save the figure here when given (format from the suffix:
            .png/.pdf/.eps/... — PyGMT-native). No default output path;
            None (default) skips saving.
        show: Open the figure in an external viewer after rendering.
        parser: Optional ``gps_parser.ConfigParser`` to reuse when
            resolving markers.

    Returns:
        The ``pygmt.Figure``, so callers can add layers or save in
        additional formats.

    Raises:
        ImportError: When PyGMT / the GMT C library is unavailable (see
            module docstring for installation).
        ValueError: When ``stations`` is empty and no ``region`` is given.
    """
    pygmt_mod = _import_pygmt()

    if not stations and region is None:
        raise ValueError("station_map needs a station or a region")
    coordinates = _resolve_stations(stations, parser) if stations else []

    if region is None:
        region = region_around(coordinates, margin=margin)

    fig = pygmt_mod.Figure()
    _draw_basemap(
        pygmt_mod,
        fig,
        region=region,
        projection=projection,
        resolution=resolution,
        land=land,
        water=water,
        shorelines=shorelines,
        dem_grid=dem_grid,
        dem_cmap=dem_cmap,
        bathymetry_cmap=bathymetry_cmap,
        glacier_cmap=glacier_cmap,
        dem_shading_azimuth=dem_shading_azimuth,
        dem_shading_normalize=dem_shading_normalize,
        dem_smooth=dem_smooth,
        overlays=overlays,
        coastline=coastline,
        glaciers=glaciers,
        lakes=lakes,
        glacier_contours=glacier_contours,
    )

    lons = [c.lon for c in coordinates]
    lats = [c.lat for c in coordinates]
    if not (markers and coordinates):
        labels = False
    elif any(getattr(c, "status", None) for c in coordinates):
        _draw_status_markers(fig, coordinates, legend_position=status_legend)
    else:
        fig.plot(
            **layer_kw(
                "stations",
                x=lons, y=lats, style=marker_style, pen=marker_pen, fill=marker_fill,
            )
        )  # fmt: skip
    if labels:
        # x=longitude, y=latitude — the dormant gmtplot.velomap had these
        # swapped in its fig.text call; fixed on absorption.
        fig.text(
            **layer_kw(
                "labels",
                x=lons,
                y=lats,
                text=[c.marker for c in coordinates],
                font=label_font,
                justify="BL",
                fill="white",
                offset=label_offset,
            )
        )

    _draw_frame(fig, _frame_args(frame, title))

    if outfile is not None:
        fig.savefig(os.fspath(outfile), show=show)
    elif show:
        fig.show()

    return fig


# =====================================================================
# Velocity map (slice 2)
# =====================================================================


@dataclasses.dataclass(frozen=True)
class VelocityVector:
    """One station's horizontal velocity with its formal uncertainty.

    The GMT ``velo`` record for one station — one row of the
    ``(longitude, latitude, east_velo, north_velo, east_sigma,
    north_sigma, correlation_EN)`` schema. σ semantics are whatever the
    estimator produced: WLS formal σ or colored-noise MLE honest σ (the
    honest ellipse is simply bigger); ``method`` carries the estimator
    tag so mixed sets can be color-coded.

    Attributes:
        marker: Station marker, e.g. ``"RHOF"``.
        lon: Geodetic longitude [decimal degrees].
        lat: Geodetic latitude [decimal degrees].
        east: East velocity [mm/yr].
        north: North velocity [mm/yr].
        sigma_east: 1-σ of ``east`` [mm/yr] (0 = no ellipse).
        sigma_north: 1-σ of ``north`` [mm/yr] (0 = no ellipse).
        correlation_en: East/north error correlation [-1, 1].
        method: Estimator tag (``"wls"`` / ``"mle"`` / ``"gbis"``) or None.
    """

    marker: str
    lon: float
    lat: float
    east: float
    north: float
    sigma_east: float = 0.0
    sigma_north: float = 0.0
    correlation_en: float = 0.0
    method: str | None = None


def velocity_vectors(
    stations: Sequence[str] | Sequence[StationCoordinate],
    east: ArrayLike,
    north: ArrayLike,
    sigma_east: ArrayLike | None = None,
    sigma_north: ArrayLike | None = None,
    correlation_en: ArrayLike | None = None,
    *,
    method: str | None = None,
    parser: Any | None = None,
) -> tuple[VelocityVector, ...]:
    """Build :class:`VelocityVector` records from velocity arrays.

    The array-boundary constructor for ``gps_analysis`` velocity output:
    per-station rates and σ (mm/yr) aligned with ``stations``; positions
    are resolved from ``stations.cfg`` via :mod:`gps_parser` (or taken
    from pre-resolved :class:`StationCoordinate` objects).

    Args:
        stations: Markers or :class:`StationCoordinate` objects, length N.
        east: East velocities [mm/yr], shape ``(N,)``.
        north: North velocities [mm/yr], shape ``(N,)``.
        sigma_east: 1-σ east [mm/yr], shape ``(N,)``; zeros when None.
        sigma_north: 1-σ north [mm/yr], shape ``(N,)``; zeros when None.
        correlation_en: E/N error correlations, shape ``(N,)``; zeros
            when None (``gps_analysis`` per-component WLS/MLE estimates
            are uncorrelated by construction).
        method: Estimator tag stamped on every record (``"wls"``,
            ``"mle"``, ...).
        parser: Optional ``gps_parser.ConfigParser`` to reuse.

    Returns:
        One :class:`VelocityVector` per station, input order preserved.

    Raises:
        ValueError: When the array lengths do not match ``stations``.
    """
    coords = _resolve_stations(stations, parser)
    n = len(coords)
    ve = np.asarray(east, dtype=np.float64)
    vn = np.asarray(north, dtype=np.float64)
    zeros = np.zeros(n, dtype=np.float64)
    se = zeros if sigma_east is None else np.asarray(sigma_east, dtype=np.float64)
    sn = zeros if sigma_north is None else np.asarray(sigma_north, dtype=np.float64)
    corr = (
        zeros
        if correlation_en is None
        else np.asarray(correlation_en, dtype=np.float64)
    )
    for label, arr in (
        ("east", ve),
        ("north", vn),
        ("sigma_east", se),
        ("sigma_north", sn),
        ("correlation_en", corr),
    ):
        if arr.shape != (n,):
            raise ValueError(
                f"{label} must have shape ({n},) matching stations, got {arr.shape}"
            )
    return tuple(
        VelocityVector(
            marker=c.marker,
            lon=c.lon,
            lat=c.lat,
            east=float(ve[i]),
            north=float(vn[i]),
            sigma_east=float(se[i]),
            sigma_north=float(sn[i]),
            correlation_en=float(corr[i]),
            method=method,
        )
        for i, c in enumerate(coords)
    )


def velocity_vectors_from_geojson(
    collection: Mapping[str, Any],
) -> tuple[VelocityVector, ...]:
    """Parse the ``gps_api`` velocities GeoJSON into velocity vectors.

    Consumes the ``GET /velocities`` product (``VelocityCollection`` —
    also the store artifact ``velocities/<region>.geojson``): a GeoJSON
    FeatureCollection whose features carry Point geometry
    ``[lon, lat(, height)]`` and ``VelocityProperties`` (``marker``,
    ``east``/``north`` [mm/yr], ``sigma_east``/``sigma_north``,
    ``method`` — ``"wls"`` formal σ or ``"mle"`` honest colored-noise σ,
    contract Amendment A5). The properties carry no E/N correlation
    (per-component estimation), so ``correlation_en`` is 0.

    Args:
        collection: The parsed GeoJSON mapping (``json.load`` output).

    Returns:
        One :class:`VelocityVector` per feature, feature order preserved.

    Raises:
        ValueError: When the mapping has no ``features`` list.
        KeyError: When a feature misses geometry or a required property.
    """
    features = collection.get("features")
    if not isinstance(features, list):
        raise ValueError(
            "expected a GeoJSON FeatureCollection with a 'features' list "
            "(the gps_api GET /velocities product)"
        )
    vectors: list[VelocityVector] = []
    for feature in features:
        geometry = feature["geometry"]
        props = feature["properties"]
        coords = geometry["coordinates"]
        vectors.append(
            VelocityVector(
                marker=str(props["marker"]),
                lon=float(coords[0]),
                lat=float(coords[1]),
                east=float(props["east"]),
                north=float(props["north"]),
                sigma_east=float(props.get("sigma_east") or 0.0),
                sigma_north=float(props.get("sigma_north") or 0.0),
                correlation_en=float(props.get("correlation_en") or 0.0),
                method=props.get("method"),
            )
        )
    return tuple(vectors)


def velocity_map(
    vectors: Sequence[VelocityVector],
    *,
    region: tuple[float, float, float, float] | None = None,
    margin: float = DEFAULT_MARGIN_DEG,
    projection: str = DEFAULT_PROJECTION,
    resolution: str = DEFAULT_RESOLUTION,
    land: str = DEFAULT_LAND,
    water: str = DEFAULT_WATER,
    shorelines: str = DEFAULT_SHORELINES,
    frame: Sequence[str] = DEFAULT_FRAME,
    title: str | None = None,
    scale: float = DEFAULT_VELOCITY_SCALE,
    confidence: float = DEFAULT_VELO_CONFIDENCE,
    color: str = DEFAULT_VELOCITY_COLOR,
    method_colors: Mapping[str, str] | None = None,
    ellipse_fill: str | None = None,
    ellipse_pen: str | None = None,
    vector_head: str = DEFAULT_VECTOR_HEAD,
    pen_width: str = DEFAULT_VECTOR_PEN_WIDTH,
    station_style: str = DEFAULT_VECTOR_STATION_STYLE,
    station_fill: str = DEFAULT_VECTOR_STATION_FILL,
    station_pen: str = DEFAULT_VECTOR_STATION_PEN,
    labels: bool = True,
    markers: bool = True,
    label_font: str = DEFAULT_LABEL_FONT,
    label_offset: str = DEFAULT_LABEL_OFFSET,
    legend: bool = True,
    scale_ref: float | None = 20.0,
    legend_font: str = DEFAULT_LEGEND_FONT,
    legend_anchor: tuple[float, float] = DEFAULT_LEGEND_ANCHOR,
    legend_line_spacing: float = DEFAULT_LEGEND_LINE_SPACING,
    dem_grid: Any | None = None,
    dem_cmap: str = DEFAULT_DEM_CMAP,
    bathymetry_cmap: str | None = None,
    glacier_cmap: str = DEFAULT_GLACIER_FILL,
    dem_shading_azimuth: float = DEFAULT_DEM_SHADING_AZIMUTH,
    dem_shading_normalize: str = DEFAULT_DEM_SHADING_NORMALIZE,
    dem_smooth: float = 0.0,
    overlays: Sequence[MapOverlay] = (),
    coastline: str | None = None,
    glaciers: str | None = None,
    lakes: str | None = None,
    glacier_contours: float | None = None,
    outfile: str | os.PathLike[str] | None = None,
    show: bool = False,
) -> pygmt.Figure:
    """Station horizontal-velocity vectors with 1-σ error ellipses (PyGMT).

    Draws the coastline (or DEM) base, one GMT ``velo`` arrow per station
    with its formal error ellipse (``confidence`` 0.39 = 1-σ), a station
    dot under each arrow, optional station labels, and a legend block
    with a scale-reference arrow. Vectors carrying different ``method``
    tags (WLS formal σ vs MLE honest σ — the honest ellipses are simply
    bigger) are drawn as separate color-coded layers via
    ``method_colors`` and listed in the legend.

    Build the input from arrays with :func:`velocity_vectors` or from the
    ``gps_api`` velocities GeoJSON with
    :func:`velocity_vectors_from_geojson`.

    Args:
        vectors: Velocity records (at least one).
        region: Map bounds ``(W, E, S, N)`` [deg]; derived from the
            vector positions + ``margin`` when None.
        margin: Padding [deg] used when ``region`` is None.
        projection: GMT projection spec.
        resolution: GSHHG shoreline resolution.
        land: Land fill color (ignored with ``dem_grid``).
        water: Water fill color.
        shorelines: Shoreline pen.
        frame: Base-map frame spec (``title`` appended as ``+t``).
        title: Optional map title.
        scale: Arrow length [cm on map] per mm/yr (GMT ``velo`` scale).
        confidence: GMT ``velo`` ellipse confidence (0.39 = 1-σ).
        color: Arrow color for vectors without a method mapping.
        method_colors: Estimator tag → color; defaults to
            :data:`DEFAULT_METHOD_COLORS`.
        ellipse_fill: Optional fill of the uncertainty ellipses (outline
            only when None).
        ellipse_pen: Outline pen of the ellipses (default: the arrow pen).
        vector_head: Arrow-head size spec.
        pen_width: Arrow/ellipse pen width.
        station_style: Station dot symbol spec.
        station_fill: Station dot fill.
        station_pen: Station dot outline pen.
        labels: Draw the station marker next to each dot.
        markers: Draw the station symbols (False: none, nor labels).
        label_font: Label font spec.
        label_offset: Label offset spec.
        legend: Draw the legend block (method entries + scale arrow).
        scale_ref: Scale-reference arrow magnitude [mm/yr]; None omits it.
        legend_font: Legend font (color appended per entry).
        legend_anchor: Legend anchor as (E-fraction, N-fraction) of the
            region from the SW corner.
        legend_line_spacing: Legend line spacing as a fraction of the N-S
            extent.
        dem_grid: Optional DEM grid for a hillshade background.
        dem_cmap: Land colours of the DEM (z >= 0): a CPT file, a
            name in the user's or gpsplot's CPT directory, a GMT CPT or a
            colour (:mod:`gps_plot.palettes`).
        bathymetry_cmap: Sea-floor colours (z < 0), named the same way;
            None = ``dem_cmap``'s sea half when it has one, else gpsplot's
            blues; ``none`` = flat ``water``, no relief.
        glacier_cmap: Glacier colours, named the same way (default white).
        dem_shading_azimuth: Hillshade illumination azimuth [deg].
        dem_shading_normalize: ``grdgradient`` normalization spec.
        dem_smooth: Smooth the hillshade (not the colours) with a
            Gaussian filter this wide [km]; 0 = off.
        overlays: GeoJSON outline layers (:class:`MapOverlay`), drawn
            over the coast/DEM and under the symbols.
        coastline: A GMT/OGR line or polygon file drawn as the shoreline
            instead of GSHHG (e.g. the IS 50V coast of the Iceland DEM).
        glaciers: A glacier polygon file: with a DEM the ice is drawn
            white with its own gentle hillshade (see :func:`_draw_basemap`).
        lakes: A lake polygon file, filled as water.
        glacier_contours: Contour interval [m] drawn on the glaciers
            (annotated every 3rd), or None.
        outfile: Save the figure here when given; None skips saving.
        show: Open the figure in an external viewer.

    Returns:
        The ``pygmt.Figure`` for further layering/saving.

    Raises:
        ImportError: When PyGMT / the GMT C library is unavailable.
        ValueError: When ``vectors`` is empty.
    """
    if not vectors:
        raise ValueError("velocity_map needs at least one velocity vector")

    lons = np.array([v.lon for v in vectors], dtype=np.float64)
    lats = np.array([v.lat for v in vectors], dtype=np.float64)
    if region is None:
        region = _region_of_points(lons.tolist(), lats.tolist(), margin)

    colors = dict(DEFAULT_METHOD_COLORS if method_colors is None else method_colors)

    pygmt_mod = _import_pygmt()

    fig = pygmt_mod.Figure()
    _draw_basemap(
        pygmt_mod,
        fig,
        region=region,
        projection=projection,
        resolution=resolution,
        land=land,
        water=water,
        shorelines=shorelines,
        dem_grid=dem_grid,
        dem_cmap=dem_cmap,
        bathymetry_cmap=bathymetry_cmap,
        glacier_cmap=glacier_cmap,
        dem_shading_azimuth=dem_shading_azimuth,
        dem_shading_normalize=dem_shading_normalize,
        dem_smooth=dem_smooth,
        overlays=overlays,
        coastline=coastline,
        glaciers=glaciers,
        lakes=lakes,
        glacier_contours=glacier_contours,
    )

    # One velo layer per estimator tag, first-appearance order preserved.
    group_order: list[str | None] = []
    groups: dict[str | None, list[int]] = {}
    for i, v in enumerate(vectors):
        if v.method not in groups:
            groups[v.method] = []
            group_order.append(v.method)
        groups[v.method].append(i)

    legend_entries: list[tuple[str, str]] = []
    for method in group_order:
        idx = np.array(groups[method], dtype=int)
        layer_color = colors.get(method, color) if method else color
        _draw_vector_layer(
            fig,
            lons[idx],
            lats[idx],
            np.array([vectors[i].east for i in idx], dtype=np.float64),
            np.array([vectors[i].north for i in idx], dtype=np.float64),
            sigma_east=np.array([vectors[i].sigma_east for i in idx], dtype=np.float64),
            sigma_north=np.array(
                [vectors[i].sigma_north for i in idx], dtype=np.float64
            ),
            correlation=np.array(
                [vectors[i].correlation_en for i in idx], dtype=np.float64
            ),
            color=layer_color,
            scale=scale,
            confidence=confidence,
            head=vector_head,
            pen_width=pen_width,
            ellipse_fill=ellipse_fill,
            ellipse_pen=ellipse_pen,
        )
        legend_entries.append((method if method else "velocity", layer_color))

    if markers:
        fig.plot(
            **layer_kw(
                "stations",
                x=lons, y=lats, style=station_style, fill=station_fill,
                pen=station_pen,
            )
        )  # fmt: skip
    if labels and markers:
        fig.text(
            **layer_kw(
                "labels",
                x=lons,
                y=lats,
                text=[v.marker for v in vectors],
                font=label_font,
                justify="BL",
                fill="white",
                offset=label_offset,
            )
        )

    if legend:
        _draw_vector_legend(
            fig,
            region=region,
            entries=legend_entries,
            scale_ref=scale_ref,
            scale_unit="mm/yr",
            scale=scale,
            confidence=confidence,
            head=vector_head,
            pen_width=pen_width,
            font=legend_font,
            anchor=legend_anchor,
            line_spacing=legend_line_spacing,
        )

    _draw_frame(fig, _frame_args(frame, title))

    if outfile is not None:
        fig.savefig(os.fspath(outfile), show=show)
    elif show:
        fig.show()

    return fig


# =====================================================================
# Observed vs model deformation vectors
# =====================================================================


@dataclasses.dataclass(frozen=True)
class VectorField:
    """One named horizontal vector field over the map's station set.

    Displacement (or velocity) components aligned with the ``stations``
    argument of :func:`deformation_vectors` — a model prediction
    (:func:`mogi_model_field` / :func:`okada_model_field`) or any
    pre-computed field. Non-finite components drop that station from the
    layer (missing observation).

    Attributes:
        name: Legend label, e.g. ``"Mogi 3.7 km"``.
        east: East components [mm], shape ``(N,)``.
        north: North components [mm], shape ``(N,)``.
        color: GMT color of the arrows (and the source marker).
        source_lon: Optional model-source longitude for a star marker.
        source_lat: Optional model-source latitude for a star marker.
    """

    name: str
    east: Any
    north: Any
    color: str = "red"
    source_lon: float | None = None
    source_lat: float | None = None


def mogi_model_field(
    lon: ArrayLike,
    lat: ArrayLike,
    *,
    source_lon: float,
    source_lat: float,
    depth: float,
    dv: float,
    name: str,
    color: str = "red",
    nu: float | None = None,
) -> VectorField:
    """Predicted horizontal Mogi displacements as a :class:`VectorField`.

    Maps the stations into a source-centered local frame
    (:func:`gps_analysis.local_coordinates`) and evaluates
    :func:`gps_analysis.mogi_forward` — the parameterization of the
    ``gps_api`` Mogi products (``DeformationResult`` fits /
    ``models/<region>_deformation.json``: source lon/lat, ``depth_km``
    · 1e3, ``dv_m3``).

    Args:
        lon: Station longitudes [deg], shape ``(N,)``.
        lat: Station latitudes [deg], shape ``(N,)``.
        source_lon: Source longitude [deg].
        source_lat: Source latitude [deg].
        depth: Source depth [m], > 0.
        dv: Volume change ΔV [m³] (+ = inflation).
        name: Legend label of the field.
        color: Arrow/source-marker color.
        nu: Poisson's ratio; ``gps_analysis`` default (0.25) when None.

    Returns:
        A :class:`VectorField` with horizontal components in mm and the
        source position attached (star marker on the map).

    Raises:
        ImportError: When :mod:`gps_analysis` is not installed.
    """
    ga = _import_gps_analysis()
    e, n = ga.local_coordinates(lon, lat, source_lon, source_lat)
    source = ga.MogiSource(x=0.0, y=0.0, depth=depth, dv=dv)
    kwargs = {} if nu is None else {"nu": nu}
    enu = ga.mogi_forward(e, n, source, **kwargs)
    return VectorField(
        name=name,
        east=enu[0] * 1e3,
        north=enu[1] * 1e3,
        color=color,
        source_lon=source_lon,
        source_lat=source_lat,
    )


def okada_model_field(
    lon: ArrayLike,
    lat: ArrayLike,
    *,
    origin_lon: float,
    origin_lat: float,
    source: Any,
    name: str,
    color: str = "red",
    nu: float | None = None,
    source_lon: float | None = None,
    source_lat: float | None = None,
) -> VectorField:
    """Predicted horizontal Okada displacements as a :class:`VectorField`.

    Maps the stations into the local tangent-plane frame anchored at
    ``(origin_lon, origin_lat)`` (:func:`gps_analysis.local_coordinates`)
    and evaluates :func:`gps_analysis.okada_forward` for the given
    ``gps_analysis.OkadaSource`` (centroid convention; ``x``/``y`` are
    metres from the origin — the frame of the ``gps_api`` Okada products,
    whose origin is the configured plane centroid).

    Args:
        lon: Station longitudes [deg], shape ``(N,)``.
        lat: Station latitudes [deg], shape ``(N,)``.
        origin_lon: Local-frame origin longitude [deg].
        origin_lat: Local-frame origin latitude [deg].
        source: A ``gps_analysis.OkadaSource`` (geometry + slip, metres).
        name: Legend label of the field.
        color: Arrow/source-marker color.
        nu: Poisson's ratio; ``gps_analysis`` default (0.25) when None.
        source_lon: Optional star-marker longitude (e.g. the origin).
        source_lat: Optional star-marker latitude.

    Returns:
        A :class:`VectorField` with horizontal components in mm.

    Raises:
        ImportError: When :mod:`gps_analysis` is not installed.
    """
    ga = _import_gps_analysis()
    e, n = ga.local_coordinates(lon, lat, origin_lon, origin_lat)
    kwargs = {} if nu is None else {"nu": nu}
    enu = ga.okada_forward(e, n, source, **kwargs)
    return VectorField(
        name=name,
        east=enu[0] * 1e3,
        north=enu[1] * 1e3,
        color=color,
        source_lon=source_lon,
        source_lat=source_lat,
    )


def deformation_vectors(
    stations: Sequence[str] | Sequence[StationCoordinate],
    observed_east: ArrayLike,
    observed_north: ArrayLike,
    models: Sequence[VectorField] = (),
    *,
    observed_label: str = "observed",
    observed_color: str = DEFAULT_OBSERVED_COLOR,
    region: tuple[float, float, float, float] | None = None,
    margin: float = DEFAULT_MARGIN_DEG,
    projection: str = DEFAULT_PROJECTION,
    resolution: str = DEFAULT_RESOLUTION,
    land: str = DEFAULT_LAND,
    water: str = DEFAULT_WATER,
    shorelines: str = DEFAULT_SHORELINES,
    frame: Sequence[str] = DEFAULT_FRAME,
    title: str | None = None,
    scale: float = DEFAULT_DISPLACEMENT_SCALE,
    confidence: float = DEFAULT_VELO_CONFIDENCE,
    vector_head: str = DEFAULT_VECTOR_HEAD,
    pen_width: str = DEFAULT_VECTOR_PEN_WIDTH,
    station_style: str = DEFAULT_VECTOR_STATION_STYLE,
    station_fill: str = DEFAULT_VECTOR_STATION_FILL,
    station_pen: str = DEFAULT_VECTOR_STATION_PEN,
    labels: bool = False,
    markers: bool = True,
    label_font: str = DEFAULT_LABEL_FONT,
    label_offset: str = DEFAULT_LABEL_OFFSET,
    source_style: str = DEFAULT_SOURCE_STYLE,
    source_pen: str = DEFAULT_SOURCE_PEN,
    legend: bool = True,
    scale_ref: float | None = 50.0,
    scale_unit: str = "mm",
    legend_font: str = DEFAULT_LEGEND_FONT,
    legend_anchor: tuple[float, float] = DEFAULT_LEGEND_ANCHOR,
    legend_line_spacing: float = DEFAULT_LEGEND_LINE_SPACING,
    dem_grid: Any | None = None,
    dem_cmap: str = DEFAULT_DEM_CMAP,
    bathymetry_cmap: str | None = None,
    glacier_cmap: str = DEFAULT_GLACIER_FILL,
    dem_shading_azimuth: float = DEFAULT_DEM_SHADING_AZIMUTH,
    dem_shading_normalize: str = DEFAULT_DEM_SHADING_NORMALIZE,
    dem_smooth: float = 0.0,
    overlays: Sequence[MapOverlay] = (),
    coastline: str | None = None,
    glaciers: str | None = None,
    lakes: str | None = None,
    glacier_contours: float | None = None,
    parser: Any | None = None,
    outfile: str | os.PathLike[str] | None = None,
    show: bool = False,
) -> pygmt.Figure:
    """Observed displacement field vs model predictions on one map (PyGMT).

    The reusable generalization of the Svartsengi Mogi-comparison figure:
    the observed horizontal displacement field plus any number of
    color-coded model vector fields (our Mogi fit, a reference model, an
    Okada prediction, ...) drawn over the coastline (or DEM) base, with
    station dots, model source markers (stars), a scale-reference arrow
    and a color-keyed legend. Model layers are drawn first in the given
    order; the observed field is drawn last, on top.

    Args:
        stations: Markers (resolved via :mod:`gps_parser`) or
            pre-resolved :class:`StationCoordinate` objects, length N.
        observed_east: Observed east displacements [mm], shape ``(N,)``;
            NaN drops that station's observed arrow.
        observed_north: Observed north displacements [mm], shape ``(N,)``.
        models: Model vector fields aligned with ``stations`` — build
            them with :func:`mogi_model_field` / :func:`okada_model_field`
            or pass pre-computed components (mm).
        observed_label: Legend label of the observed field.
        observed_color: Arrow color of the observed field.
        region: Map bounds ``(W, E, S, N)`` [deg]; derived from the
            station bounding box + ``margin`` when None.
        margin: Padding [deg] used when ``region`` is None.
        projection: GMT projection spec.
        resolution: GSHHG shoreline resolution.
        land: Land fill color (ignored with ``dem_grid``).
        water: Water fill color.
        shorelines: Shoreline pen.
        frame: Base-map frame spec (``title`` appended as ``+t``).
        title: Optional map title.
        scale: Arrow length [cm on map] per mm.
        confidence: GMT ``velo`` ellipse confidence (unused while the
            displacement fields carry no σ, but kept uniform).
        vector_head: Arrow-head size spec.
        pen_width: Arrow pen width.
        station_style: Station dot symbol spec.
        station_fill: Station dot fill.
        station_pen: Station dot outline pen.
        labels: Draw station markers next to the dots.
        markers: Draw the station symbols (False: none, nor labels).
        label_font: Label font spec.
        label_offset: Label offset spec.
        source_style: Symbol spec of the model source markers.
        source_pen: Outline pen of the source markers.
        legend: Draw the legend block.
        scale_ref: Scale-reference arrow magnitude [``scale_unit``];
            None omits it.
        scale_unit: Unit label of the reference arrow (default mm).
        legend_font: Legend font (color appended per entry).
        legend_anchor: Legend anchor (region fractions from SW corner).
        legend_line_spacing: Legend line spacing (fraction of N-S extent).
        dem_grid: Optional DEM grid for a hillshade background.
        dem_cmap: Land colours of the DEM (z >= 0): a CPT file, a
            name in the user's or gpsplot's CPT directory, a GMT CPT or a
            colour (:mod:`gps_plot.palettes`).
        bathymetry_cmap: Sea-floor colours (z < 0), named the same way;
            None = ``dem_cmap``'s sea half when it has one, else gpsplot's
            blues; ``none`` = flat ``water``, no relief.
        glacier_cmap: Glacier colours, named the same way (default white).
        dem_shading_azimuth: Hillshade illumination azimuth [deg].
        dem_shading_normalize: ``grdgradient`` normalization spec.
        dem_smooth: Smooth the hillshade (not the colours) with a
            Gaussian filter this wide [km]; 0 = off.
        overlays: GeoJSON outline layers (:class:`MapOverlay`), drawn
            over the coast/DEM and under the symbols.
        coastline: A GMT/OGR line or polygon file drawn as the shoreline
            instead of GSHHG (e.g. the IS 50V coast of the Iceland DEM).
        glaciers: A glacier polygon file: with a DEM the ice is drawn
            white with its own gentle hillshade (see :func:`_draw_basemap`).
        lakes: A lake polygon file, filled as water.
        glacier_contours: Contour interval [m] drawn on the glaciers
            (annotated every 3rd), or None.
        parser: Optional ``gps_parser.ConfigParser`` to reuse.
        outfile: Save the figure here when given; None skips saving.
        show: Open the figure in an external viewer.

    Returns:
        The ``pygmt.Figure`` for further layering/saving.

    Raises:
        ImportError: When PyGMT / the GMT C library is unavailable.
        ValueError: When ``stations`` is empty or an array/field length
            does not match it.
    """
    if not stations:
        raise ValueError("deformation_vectors needs at least one station")
    coordinates = _resolve_stations(stations, parser)
    n = len(coordinates)
    lons = np.array([c.lon for c in coordinates], dtype=np.float64)
    lats = np.array([c.lat for c in coordinates], dtype=np.float64)

    obs_e = np.asarray(observed_east, dtype=np.float64)
    obs_n = np.asarray(observed_north, dtype=np.float64)
    if obs_e.shape != (n,) or obs_n.shape != (n,):
        raise ValueError(
            f"observed components must have shape ({n},) matching stations, "
            f"got {obs_e.shape} / {obs_n.shape}"
        )
    fields: list[tuple[VectorField, Any, Any]] = []
    for field in models:
        fe = np.asarray(field.east, dtype=np.float64)
        fn = np.asarray(field.north, dtype=np.float64)
        if fe.shape != (n,) or fn.shape != (n,):
            raise ValueError(
                f"model field {field.name!r} must have shape ({n},) matching "
                f"stations, got {fe.shape} / {fn.shape}"
            )
        fields.append((field, fe, fn))

    if region is None:
        region = region_around(coordinates, margin=margin)

    pygmt_mod = _import_pygmt()
    fig = pygmt_mod.Figure()
    _draw_basemap(
        pygmt_mod,
        fig,
        region=region,
        projection=projection,
        resolution=resolution,
        land=land,
        water=water,
        shorelines=shorelines,
        dem_grid=dem_grid,
        dem_cmap=dem_cmap,
        bathymetry_cmap=bathymetry_cmap,
        glacier_cmap=glacier_cmap,
        dem_shading_azimuth=dem_shading_azimuth,
        dem_shading_normalize=dem_shading_normalize,
        dem_smooth=dem_smooth,
        overlays=overlays,
        coastline=coastline,
        glaciers=glaciers,
        lakes=lakes,
        glacier_contours=glacier_contours,
    )

    for field, fe, fn in fields:
        _draw_vector_layer(
            fig,
            lons,
            lats,
            fe,
            fn,
            color=field.color,
            scale=scale,
            confidence=confidence,
            head=vector_head,
            pen_width=pen_width,
        )
    _draw_vector_layer(
        fig,
        lons,
        lats,
        obs_e,
        obs_n,
        color=observed_color,
        scale=scale,
        confidence=confidence,
        head=vector_head,
        pen_width=pen_width,
    )

    if markers:
        fig.plot(
            **layer_kw(
                "stations",
                x=lons, y=lats, style=station_style, fill=station_fill,
                pen=station_pen,
            )
        )  # fmt: skip
    if labels and markers:
        fig.text(
            **layer_kw(
                "labels",
                x=lons,
                y=lats,
                text=[c.marker for c in coordinates],
                font=label_font,
                justify="BL",
                fill="white",
                offset=label_offset,
            )
        )
    for field, _, _ in fields:
        if field.source_lon is not None and field.source_lat is not None:
            fig.plot(
                x=[field.source_lon],
                y=[field.source_lat],
                style=source_style,
                fill=field.color,
                pen=source_pen,
            )

    if legend:
        entries = [(observed_label, observed_color)]
        entries += [(field.name, field.color) for field, _, _ in fields]
        _draw_vector_legend(
            fig,
            region=region,
            entries=entries,
            scale_ref=scale_ref,
            scale_unit=scale_unit,
            scale=scale,
            confidence=confidence,
            head=vector_head,
            pen_width=pen_width,
            font=legend_font,
            anchor=legend_anchor,
            line_spacing=legend_line_spacing,
        )

    _draw_frame(fig, _frame_args(frame, title))

    if outfile is not None:
        fig.savefig(os.fspath(outfile), show=show)
    elif show:
        fig.show()

    return fig


# =====================================================================
# Okada distributed-slip map
# =====================================================================


@dataclasses.dataclass(frozen=True)
class SlipPatch:
    """One fault-patch polygon with its slip/opening value.

    Coordinate meaning depends on the view: map view uses geodetic
    longitude/latitude corners (surface projection of the patch); plane
    view uses along-strike / down-dip distances [km] on the fault plane.

    Attributes:
        xs: Polygon corner x-coordinates (lon [deg] or along-strike [km]).
        ys: Polygon corner y-coordinates (lat [deg] or down-dip [km]).
        value: Slip/opening magnitude of the patch [m].
    """

    xs: tuple[float, ...]
    ys: tuple[float, ...]
    value: float


def _resolve_slip_component(product: Mapping[str, Any], component: str | None) -> str:
    """The slip component to color by: explicit, or largest |potency|."""
    patches = product.get("patches") or []
    if not patches:
        raise ValueError("slip product carries no patches")
    # Annotated because `product` is Mapping[str, Any]: without it the key
    # type stays Any all the way through `totals`, and both `max(...)`
    # returns below silently widen this function's declared `str`.
    available: set[str] = set(patches[0]["slip_m"])
    if component is not None:
        if component not in available:
            raise ValueError(
                f"component {component!r} not in the product's slip "
                f"components {sorted(available)}"
            )
        return component
    potency: Mapping[str, Any] | None = product.get("potency_m3")
    if potency:
        return max(potency, key=lambda k: abs(float(potency[k])))
    totals = {c: sum(abs(float(p["slip_m"][c])) for p in patches) for c in available}
    return max(totals, key=lambda k: totals[k])


def slip_patches_from_product(
    product: Mapping[str, Any],
    *,
    component: str | None = None,
) -> tuple[tuple[SlipPatch, ...], str]:
    """Map-view patch polygons from a ``SlipDistributionResult`` product.

    Consumes the ``gps_api`` Okada distributed-slip product
    (``GET /v1/deformation/{region}`` with ``source_type="okada"``, the
    store artifact ``models/<region>_slip.json``): per-patch centroids
    (``lon``/``lat``) plus the plane geometry (``strike``, ``dip``,
    ``length_km``/``n_strike``, ``width_km``/``n_dip``) are turned into
    the surface-projected corner polygon of each patch. Corner offsets
    are computed in the local tangent-plane frame (patch length along
    strike, ``width·cos(dip)`` along the dip direction) and converted to
    degrees with the metric of :func:`gps_analysis.local_coordinates`
    at the product origin — no projection math is re-derived here.

    Note: for a near-vertical plane (dip → 90°) the surface projection
    degenerates toward a line along strike; use ``view="plane"`` in
    :func:`slip_map` for the readable cross-section.

    Args:
        product: The parsed product mapping (``json.load`` output or the
            model dump of a ``gps_api.schemas.SlipDistributionResult``).
        component: Slip component to color by (``"opening"`` /
            ``"strike_slip"`` / ``"dip_slip"``); when None the component
            with the largest |potency| is used.

    Returns:
        ``(patches, component_used)`` — one :class:`SlipPatch` per product
        patch (product order preserved) and the component the values
        belong to.

    Raises:
        ImportError: When :mod:`gps_analysis` is not installed.
        ValueError: When the product has no patches or the component is
            unknown.
    """
    comp = _resolve_slip_component(product, component)
    ga = _import_gps_analysis()

    origin_lon = float(product["origin_lon"])
    origin_lat = float(product["origin_lat"])
    step = 1.0e-3  # deg; local_coordinates is linear, so the step cancels
    de, _ = ga.local_coordinates(origin_lon + step, origin_lat, origin_lon, origin_lat)
    _, dn = ga.local_coordinates(origin_lon, origin_lat + step, origin_lon, origin_lat)
    m_per_deg_lon = float(np.asarray(de)) / step
    m_per_deg_lat = float(np.asarray(dn)) / step

    strike = math.radians(float(product["strike"]))
    dip = math.radians(float(product["dip"]))
    patch_length = float(product["length_km"]) * 1.0e3 / int(product["n_strike"])
    patch_width = float(product["width_km"]) * 1.0e3 / int(product["n_dip"])
    half_l = 0.5 * patch_length
    half_w = 0.5 * patch_width * math.cos(dip)  # horizontal down-dip extent
    u_strike = (math.sin(strike), math.cos(strike))
    u_dip = (math.cos(strike), -math.sin(strike))  # 90 deg clockwise of strike
    corner_offsets = (
        (-half_l, -half_w),
        (half_l, -half_w),
        (half_l, half_w),
        (-half_l, half_w),
    )

    patches: list[SlipPatch] = []
    for patch in product["patches"]:
        lon0 = float(patch["lon"])
        lat0 = float(patch["lat"])
        xs = tuple(
            lon0 + (ds * u_strike[0] + dw * u_dip[0]) / m_per_deg_lon
            for ds, dw in corner_offsets
        )
        ys = tuple(
            lat0 + (ds * u_strike[1] + dw * u_dip[1]) / m_per_deg_lat
            for ds, dw in corner_offsets
        )
        patches.append(SlipPatch(xs=xs, ys=ys, value=float(patch["slip_m"][comp])))
    return tuple(patches), comp


def _plane_patches(product: Mapping[str, Any], component: str) -> tuple[SlipPatch, ...]:
    """Fault-plane cross-section rectangles (along-strike/down-dip km)."""
    patch_length = float(product["length_km"]) / int(product["n_strike"])
    patch_width = float(product["width_km"]) / int(product["n_dip"])
    patches: list[SlipPatch] = []
    for patch in product["patches"]:
        col = int(patch["col"])
        row = int(patch["row"])
        x0, x1 = col * patch_length, (col + 1) * patch_length
        y0, y1 = row * patch_width, (row + 1) * patch_width
        patches.append(
            SlipPatch(
                xs=(x0, x1, x1, x0),
                ys=(y0, y0, y1, y1),
                value=float(patch["slip_m"][component]),
            )
        )
    return tuple(patches)


def slip_map(
    slip: Mapping[str, Any] | Sequence[SlipPatch],
    *,
    component: str | None = None,
    view: str = "map",
    cmap: str = DEFAULT_SLIP_CMAP,
    cmap_reverse: bool = DEFAULT_SLIP_CMAP_REVERSE,
    series: tuple[float, float] | None = None,
    region: tuple[float, float, float, float] | None = None,
    margin: float = DEFAULT_MARGIN_DEG,
    projection: str = DEFAULT_PROJECTION,
    resolution: str = DEFAULT_RESOLUTION,
    land: str = DEFAULT_LAND,
    water: str = DEFAULT_WATER,
    shorelines: str = DEFAULT_SHORELINES,
    frame: Sequence[str] = DEFAULT_FRAME,
    title: str | None = None,
    patch_pen: str = DEFAULT_PATCH_PEN,
    stations: Sequence[str] | Sequence[StationCoordinate] | None = None,
    marker_style: str = DEFAULT_VECTOR_STATION_STYLE,
    marker_pen: str = DEFAULT_VECTOR_STATION_PEN,
    marker_fill: str = DEFAULT_VECTOR_STATION_FILL,
    labels: bool = True,
    label_font: str = DEFAULT_LABEL_FONT,
    label_offset: str = DEFAULT_LABEL_OFFSET,
    colorbar: bool = True,
    colorbar_label: str | None = None,
    plane_projection: str = DEFAULT_PLANE_PROJECTION,
    plane_xlabel: str = DEFAULT_PLANE_XLABEL,
    plane_ylabel: str = DEFAULT_PLANE_YLABEL,
    dem_grid: Any | None = None,
    dem_cmap: str = DEFAULT_DEM_CMAP,
    bathymetry_cmap: str | None = None,
    glacier_cmap: str = DEFAULT_GLACIER_FILL,
    dem_shading_azimuth: float = DEFAULT_DEM_SHADING_AZIMUTH,
    dem_shading_normalize: str = DEFAULT_DEM_SHADING_NORMALIZE,
    dem_smooth: float = 0.0,
    overlays: Sequence[MapOverlay] = (),
    coastline: str | None = None,
    glaciers: str | None = None,
    lakes: str | None = None,
    glacier_contours: float | None = None,
    parser: Any | None = None,
    outfile: str | os.PathLike[str] | None = None,
    show: bool = False,
) -> pygmt.Figure:
    """Okada distributed-slip distribution as a colored patch map (PyGMT).

    Draws each fault patch as a polygon colored by its slip/opening
    magnitude with a colorbar. Two views:

    * ``view="map"`` (default): surface-projected patch polygons on the
      coastline (or DEM) base map, optionally with station markers.
    * ``view="plane"``: the fault-plane cross-section (along-strike vs
      down-dip distance [km], depth increasing downward) — the readable
      view for near-vertical dikes, whose map-view projection collapses
      toward a line.

    Consumes the ``gps_api`` ``SlipDistributionResult`` product directly
    (a mapping — per-patch geometry is derived via
    :func:`slip_patches_from_product` / the plane grid), or pre-built
    :class:`SlipPatch` polygons for full control (arrays at the API
    boundary; ``component`` then only labels the colorbar).

    Args:
        slip: The parsed product mapping, or a sequence of
            :class:`SlipPatch` (map view: lon/lat corners; plane view:
            km corners).
        component: Slip component to color by; largest-|potency|
            component of the product when None.
        view: ``"map"`` or ``"plane"``.
        cmap: Sequential colour map for the slip values.
        cmap_reverse: Reverse the colour map (default True: GMT's
            lajolla runs dark to light, and zero slip should be light).
        series: Explicit ``(min, max)`` of the colour scale; derived
            from the values (floored at 0 for non-negative slip) when
            None.
        region: Map bounds ``(W, E, S, N)``; derived from the patch
            extent + ``margin`` (map view) or the plane size (plane
            view) when None.
        margin: Padding [deg] used when ``region`` is None (map view).
        projection: GMT projection spec (map view).
        resolution: GSHHG shoreline resolution (map view).
        land: Land fill color (map view; ignored with ``dem_grid``).
        water: Water fill color (map view).
        shorelines: Shoreline pen (map view).
        frame: Base-map frame spec (``title`` appended as ``+t``).
        title: Optional title.
        patch_pen: Patch outline pen.
        stations: Optional stations to overlay (map view only) — markers
            or :class:`StationCoordinate` objects.
        marker_style: Station symbol spec.
        marker_pen: Station outline pen.
        marker_fill: Station fill.
        labels: Label the overlaid stations.
        label_font: Label font spec.
        label_offset: Label offset spec.
        colorbar: Draw the colorbar.
        colorbar_label: Colorbar label; ``"<component> slip (m)"`` when
            None.
        plane_projection: Cartesian projection of the plane view
            (negative height flips the down-dip axis downward).
        plane_xlabel: Plane-view x-axis label.
        plane_ylabel: Plane-view y-axis label.
        dem_grid: Optional DEM grid for a hillshade background (map view).
        dem_cmap: Land colours of the DEM (z >= 0): a CPT file, a
            name in the user's or gpsplot's CPT directory, a GMT CPT or a
            colour (:mod:`gps_plot.palettes`).
        bathymetry_cmap: Sea-floor colours (z < 0), named the same way;
            None = ``dem_cmap``'s sea half when it has one, else gpsplot's
            blues; ``none`` = flat ``water``, no relief.
        glacier_cmap: Glacier colours, named the same way (default white).
        dem_shading_azimuth: Hillshade illumination azimuth [deg].
        dem_shading_normalize: ``grdgradient`` normalization spec.
        dem_smooth: Smooth the hillshade (not the colours) with a
            Gaussian filter this wide [km]; 0 = off.
        overlays: GeoJSON outline layers (:class:`MapOverlay`), drawn
            over the coast/DEM and under the symbols.
        coastline: A GMT/OGR line or polygon file drawn as the shoreline
            instead of GSHHG (e.g. the IS 50V coast of the Iceland DEM).
        glaciers: A glacier polygon file: with a DEM the ice is drawn
            white with its own gentle hillshade (see :func:`_draw_basemap`).
        lakes: A lake polygon file, filled as water.
        glacier_contours: Contour interval [m] drawn on the glaciers
            (annotated every 3rd), or None.
        parser: Optional ``gps_parser.ConfigParser`` for station markers.
        outfile: Save the figure here when given; None skips saving.
        show: Open the figure in an external viewer.

    Returns:
        The ``pygmt.Figure`` for further layering/saving.

    Raises:
        ImportError: When PyGMT / the GMT C library is unavailable (and,
            for product input in map view, when :mod:`gps_analysis` is
            not installed).
        ValueError: When there are no patches, the component is unknown,
            or ``view`` is not ``"map"``/``"plane"``.
    """
    if view not in ("map", "plane"):
        raise ValueError(f"view must be 'map' or 'plane', got {view!r}")

    plane_region: tuple[float, float, float, float] | None = None
    if isinstance(slip, Mapping):
        comp = _resolve_slip_component(slip, component)
        if view == "map":
            patches, comp = slip_patches_from_product(slip, component=comp)
        else:
            patches = _plane_patches(slip, comp)
            plane_region = (
                0.0,
                float(slip["length_km"]),
                0.0,
                float(slip["width_km"]),
            )
    else:
        patches = tuple(slip)
        comp = component if component is not None else "slip"
    if not patches:
        raise ValueError("slip_map needs at least one fault patch")

    values = np.array([p.value for p in patches], dtype=np.float64)
    if series is None:
        vmin = min(0.0, float(values.min()))
        vmax = float(values.max())
        if not vmax > vmin:
            vmin, vmax = vmin - 0.5, vmax + 0.5  # constant field: pad the scale
        series = (vmin, vmax)

    all_x = [x for p in patches for x in p.xs]
    all_y = [y for p in patches for y in p.ys]

    pygmt_mod = _import_pygmt()
    fig = pygmt_mod.Figure()
    if view == "map":
        if region is None:
            region = _region_of_points(all_x, all_y, margin)
        _draw_basemap(
            pygmt_mod,
            fig,
            region=region,
            projection=projection,
            resolution=resolution,
            land=land,
            water=water,
            shorelines=shorelines,
            dem_grid=dem_grid,
            dem_cmap=dem_cmap,
            bathymetry_cmap=bathymetry_cmap,
            glacier_cmap=glacier_cmap,
            dem_shading_azimuth=dem_shading_azimuth,
            dem_shading_normalize=dem_shading_normalize,
            dem_smooth=dem_smooth,
            overlays=overlays,
            coastline=coastline,
            glaciers=glaciers,
            lakes=lakes,
            glacier_contours=glacier_contours,
        )
    else:
        if region is None:
            region = plane_region or _region_of_points(all_x, all_y, 0.0)
        plane_frame = list(frame) + [f"x+l{plane_xlabel}", f"y+l{plane_ylabel}"]
        fig.basemap(
            region=region,
            projection=plane_projection,
            frame=_frame_args(plane_frame, title),
        )

    pygmt_mod.makecpt(cmap=cmap, series=list(series), reverse=cmap_reverse)
    for patch in patches:
        fig.plot(
            x=list(patch.xs),
            y=list(patch.ys),
            close=True,
            fill="+z",
            zvalue=patch.value,
            cmap=True,
            pen=patch_pen,
        )

    if stations is not None and view == "map":
        coords = _resolve_stations(stations, parser)
        st_lons = [c.lon for c in coords]
        st_lats = [c.lat for c in coords]
        fig.plot(
            x=st_lons, y=st_lats, style=marker_style, pen=marker_pen, fill=marker_fill
        )
        if labels:
            fig.text(
                x=st_lons,
                y=st_lats,
                text=[c.marker for c in coords],
                font=label_font,
                justify="BL",
                fill="white",
                offset=label_offset,
            )

    if colorbar:
        label = colorbar_label if colorbar_label is not None else f"{comp} slip (m)"
        fig.colorbar(frame=["af", f"x+l{label}"])

    if view == "map":
        _draw_frame(fig, _frame_args(frame, title))

    if outfile is not None:
        fig.savefig(os.fspath(outfile), show=show)
    elif show:
        fig.show()

    return fig
