"""PyGMT station maps for the GNSS network (map lane, PLAN Phase 3).

First slice of the ``gps_plot`` map lane: :func:`station_map` renders a
Mercator coastline base map (GMT/GSHHG shoreline database) with GNSS
station markers and optional name labels for an arbitrary region.

What the map shows
    Station positions of the IMO GNSS network (or any subset) as diamond
    markers on a coastline map. Geographic coordinates are geodetic
    longitude/latitude (WGS84/ITRF as recorded in ``stations.cfg``);
    the default projection is Mercator (GMT ``M<width>``), suitable for
    the network's regional (Iceland-scale) extents.

Data provenance
    Station coordinates and display names come from ``stations.cfg`` via
    the :mod:`gps_parser` API (``ConfigParser.getStationInfo``) — the same
    configuration source the rest of the pipeline uses (``receivers``,
    ``gps_api.precompute``). Nothing is hardcoded here: region bounds,
    projection, shoreline resolution, marker style and the output path are
    all parameters (module-level ``DEFAULT_*`` constants provide the
    defaults).

Optional dependency
    PyGMT (and the system GMT >= 6 C library underneath it) is an optional
    extra — importing :mod:`gps_plot` or this module works without it;
    only *calling* :func:`station_map` requires it. Install with::

        uv sync --extra maps        # or: pip install 'gps_plot[maps]'

    and make sure the GMT shared library is discoverable (system install,
    conda-forge ``gmt``, or ``GMT_LIBRARY_PATH`` for a from-source build).

Seam for the next slices (do not remove)
    * ``velocity_map`` (slice 2) will reuse :func:`station_map` as its base
      layer: call it with ``outfile=None``, add one ``fig.velo(...)`` call
      per velocity layer (data in the GMT *velo* schema: ``longitude,
      latitude, east_velo, north_velo, east_sigma, north_sigma,
      correlation_EN, station`` — produced from ``gps_analysis`` WLS/GBIS
      velocity output), then save. The dormant recipe lives in
      ``gps_plot.gmtplot.velomap`` until it is absorbed.
    * DEM/hillshade backgrounds (slice 3) replace the ``coast`` land fill
      with ``grdimage`` + ``grdgradient`` — again see ``gmtplot.velomap``.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, pygmt stays optional
    import pygmt

#: Default map style — parameters of :func:`station_map`, never literals in
#: the function body. Override per call; change here to restyle globally.
DEFAULT_PROJECTION = "M12c"  # Mercator, 12 cm wide
DEFAULT_RESOLUTION = "auto"  # GSHHG shoreline resolution (auto per region)
DEFAULT_LAND = "lightgrey"
DEFAULT_WATER = "lightblue"
DEFAULT_SHORELINES = "0.2p,black"
DEFAULT_FRAME: tuple[str, ...] = ("af",)
DEFAULT_MARKER_STYLE = "d7p"  # diamond, 7 points — gmtplot.velomap heritage
DEFAULT_MARKER_PEN = "1p,red"
DEFAULT_MARKER_FILL = "yellow"
DEFAULT_LABEL_FONT = "5p,Helvetica-Bold"
DEFAULT_LABEL_OFFSET = "J0.2"
#: Padding [degrees] added around the station bounding box when no explicit
#: ``region`` is given.
DEFAULT_MARGIN_DEG = 0.5


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


@dataclasses.dataclass(frozen=True)
class StationCoordinate:
    """One station's map position from ``stations.cfg``.

    Attributes:
        marker: 4-character station marker, e.g. ``"RHOF"``.
        lon: Geodetic longitude [decimal degrees, east positive].
        lat: Geodetic latitude [decimal degrees, north positive].
        name: Human-readable station name (``station_name``), or None.
    """

    marker: str
    lon: float
    lat: float
    name: str | None = None


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
    label_font: str = DEFAULT_LABEL_FONT,
    label_offset: str = DEFAULT_LABEL_OFFSET,
    outfile: str | os.PathLike[str] | None = None,
    show: bool = False,
    parser: Any | None = None,
) -> pygmt.Figure:
    """Render a coastline map with GNSS station markers (PyGMT).

    Draws a GSHHG coastline base map in the given projection, then one
    marker per station and, optionally, the station marker text next to
    it. This is the map-lane base layer: ``velocity_map`` (next slice)
    adds ``fig.velo`` arrow layers on the returned figure before saving.

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
        land: Land fill color.
        water: Water fill color.
        shorelines: Shoreline pen.
        frame: Base-map frame spec; ``title`` is appended as ``+t`` when
            given.
        title: Optional map title.
        marker_style: GMT symbol spec for the station markers.
        marker_pen: Marker outline pen.
        marker_fill: Marker fill color.
        labels: Draw the station marker next to each symbol.
        label_font: Label font spec.
        label_offset: Label offset spec (GMT ``-D`` justify form).
        outfile: Save the figure here when given (format from the suffix:
            .png/.pdf/.eps/... — PyGMT-native). No default output path;
            None (default) skips saving.
        show: Open the figure in an external viewer after rendering.
        parser: Optional ``gps_parser.ConfigParser`` to reuse when
            resolving markers.

    Returns:
        The ``pygmt.Figure``, so callers (and the future ``velocity_map``)
        can add layers or save in additional formats.

    Raises:
        ImportError: When PyGMT / the GMT C library is unavailable (see
            module docstring for installation).
        ValueError: When ``stations`` is empty.
    """
    pygmt_mod = _import_pygmt()

    if not stations:
        raise ValueError("station_map needs at least one station")
    if isinstance(stations[0], StationCoordinate):
        coordinates = tuple(stations)  # type: ignore[arg-type]
    else:
        coordinates = station_coordinates([str(s) for s in stations], parser=parser)

    if region is None:
        region = region_around(coordinates, margin=margin)

    frame_args = list(frame)
    if title is not None:
        frame_args.append(f"+t{title}")

    fig = pygmt_mod.Figure()
    fig.coast(
        region=region,
        projection=projection,
        resolution=resolution,
        land=land,
        water=water,
        shorelines=shorelines,
        frame=frame_args,
    )

    lons = [c.lon for c in coordinates]
    lats = [c.lat for c in coordinates]
    fig.plot(x=lons, y=lats, style=marker_style, pen=marker_pen, fill=marker_fill)
    if labels:
        # x=longitude, y=latitude — the dormant gmtplot.velomap had these
        # swapped in its fig.text call; fixed on absorption.
        fig.text(
            x=lons,
            y=lats,
            text=[c.marker for c in coordinates],
            font=label_font,
            justify="BL",
            fill="white",
            offset=label_offset,
        )

    if outfile is not None:
        fig.savefig(os.fspath(outfile), show=show)
    elif show:
        fig.show()

    return fig
