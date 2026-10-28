"""Named map layers: ``gpsplot map --layers lavas,barriers,…``.

A layer is a file under the local GIS directory (``gis_dir`` in
postprocess.cfg, default ``~/gps-data/gis``) with its own look and its own
``--gmt`` drawing step, so ``--gmt roads:pen=0.8p,black`` restyles the
roads alone. ``--overlay FILE`` stays for anything ad hoc.

The catalogue is configuration: ``map_layers.yaml`` in the gpsconfig
directory (or ``gpsplot map --layers-file``) adds layers and presets and
changes the built-in ones below (:func:`load_catalogue`), so a new layer or
a new look needs no code. Each layer is also a ``--gmt`` drawing step of
its own, and may carry GMT arguments of its own (``gmt:``).

The built-in Reykjanes layers are built by ``vector/build_08_reykjanes.sh`` (lavas:
the LMÍ/NÍ WFS, saved beside it; places: authored); the national IS 50V
roads, buildings and farm names by ``vector/build_09_is50v.sh``; the
volcanic systems and caldera rims (FutureVolc) by ``build_07_volcanic.sh``. A preset names
several layers in drawing order, bottom first: roads under the lavas that
cut them, the barriers over the lavas they hold back, names last.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping, Sequence
from typing import Any


@dataclasses.dataclass(frozen=True)
class GisLayer:
    """One named layer: ``path`` under the GIS directory, drawn as a
    :class:`gps_plot.maps.MapOverlay` with these keywords."""

    name: str
    path: str
    help: str
    pen: str | None = None
    fill: str | None = None
    label: str | None = None
    font: str | None = None
    symbol: str | None = None  # points: a GMT style
    #: the script that builds the file, for the missing-file hint
    build: str = "vector/build_08_reykjanes.sh"
    #: further PyGMT arguments of its drawing call (``transparency``, a
    #: GMT letter, …); ``--gmt NAME:KEY=VALUE`` still wins
    gmt: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def step(self) -> str:
        """Its ``--gmt`` drawing step: its name (a name file: the shared
        ``place-labels``)."""
        return "place-labels" if self.path.endswith(".txt") else self.name


#: drawing order is the order of a preset, not of this table
LAYERS: dict[str, GisLayer] = {
    layer.name: layer
    for layer in (
        GisLayer(
            "volcanoes", "vector/futurevolc_central_volcanoes.geojson",
            "central volcanoes, outlined and named (FutureVolc)",
            pen="0.9p,110/60/20,6_3:0", label="system",
            font="7p,Helvetica-BoldOblique,110/60/20=~0.75p,white",
            build="vector/build_07_volcanic.sh",
        ),
        GisLayer(
            "calderas", "vector/futurevolc_caldera_rims.geojson",
            "caldera rims (FutureVolc)", pen="1.2p,110/60/20",
            build="vector/build_07_volcanic.sh",
        ),
        GisLayer(
            "swarms", "vector/futurevolc_fissure_swarms.geojson",
            "fissure swarms, outlined (FutureVolc)", pen="0.6p,110/60/20,2_2:0",
            build="vector/build_07_volcanic.sh",
        ),
        GisLayer(
            "roads", "vector/is50v/roads.geojson",
            "main and connecting roads (IS 50V)", pen="0.6p,90/90/90",
            build="vector/build_09_is50v.sh",
        ),
        GisLayer(
            "localroads", "vector/is50v/localroads.geojson",
            "local roads to the farms (IS 50V héraðsvegir)", pen="0.4p,110/110/110",
            build="vector/build_09_is50v.sh",
        ),
        GisLayer(
            "buildings", "vector/is50v/buildings.geojson",
            "buildings: farms, houses, huts (IS 50V)", symbol="s0.07c",
            fill="black", pen="0.15p,white", build="vector/build_09_is50v.sh",
        ),
        GisLayer(
            "farms", "vector/is50v/farms.txt",
            "farm names (IS 50V)", font="6p,Helvetica,gray10=~0.75p,white",
            build="vector/build_09_is50v.sh",
        ),
        GisLayer(
            "towns", "vector/reykjanes/towns.geojson",
            "urban areas, named (IS 50V)", pen="0.25p,gray35",
            fill="gray55@50", label="nafnfitju",
            font="6p,Helvetica-Bold,gray15=~0.75p,white",
        ),
        GisLayer(
            "lavas", "vector/reykjanes_lavas_2021-2025_lmi_ni.geojson",
            "the 2021-2025 lavas (LMÍ/NÍ)", pen="0.25p,darkred",
            fill="orange@45", build="the LMÍ/NÍ WFS (see the GIS README)",
        ),
        GisLayer(
            "graben", "vector/reykjanes/graben.geojson",
            "the Grindavík graben, Nov 2023 and Jan 2024 (LMÍ/NÍ)",
            pen="0.5p,black,4_2:0",
        ),
        GisLayer(
            "fissures", "vector/reykjanes/fissures.geojson",
            "eruptive fissures 2021-2025 (LMÍ/NÍ)", pen="1.0p,red3",
        ),
        GisLayer(
            "barriers", "vector/reykjanes/barriers.geojson",
            "lava barriers, Svartsengi and Grindavík (IMO)", pen="1.3p,black",
        ),
        GisLayer(
            "places", "vector/reykjanes/places.txt",
            "place names: Svartsengi, Bláa lónið, … (authored)",
        ),
    )
}  # fmt: skip

PRESETS: dict[str, tuple[str, ...]] = {
    "reykjanes": ("roads", "towns", "lavas", "graben", "fissures", "barriers",
                  "places"),
    "infrastructure": ("roads", "towns", "barriers", "places"),
    "eruptions": ("lavas", "graben", "fissures"),
    # any area: the volcanic system, roads and houses (BGÓ 2026-10-28)
    "volcanic": ("swarms", "volcanoes", "calderas"),
    "rural": ("localroads", "roads", "buildings", "farms"),
}  # fmt: skip

#: ``map_layers.yaml`` keys of a layer (``file`` is its path under the GIS
#: directory, or an absolute one)
LAYER_KEYS = ("file", "help", "pen", "fill", "label", "font", "symbol", "build",
              "gmt")  # fmt: skip
#: the catalogue file in the gpsconfig directory
CATALOGUE_FILE = "map_layers.yaml"


@dataclasses.dataclass(frozen=True)
class Catalogue:
    """The layers and presets a map can name, and where they came from."""

    layers: Mapping[str, GisLayer]
    presets: Mapping[str, tuple[str, ...]]
    source: str = "built-in"


BUILTIN = Catalogue(LAYERS, PRESETS)


def _none(value: Any) -> Any:
    """``none``/``null`` in the file: no pen, no fill."""
    return None if value is None or str(value).lower() == "none" else str(value)


def catalogue_from(doc: Mapping[str, Any], source: str = "") -> Catalogue:
    """The built-in catalogue with ``doc`` (``layers:``, ``presets:``) over
    it: a layer named there replaces the keys it gives of a built-in one, or
    is new (then ``file:`` is needed); a preset replaces one of its name.

    Raises:
        ValueError: an unknown key, a new layer without a file, or a preset
            naming no layer.
    """
    unknown = set(doc) - {"layers", "presets"}
    if unknown:
        raise ValueError(f"{source}: unknown key(s) {sorted(unknown)}")
    layers = dict(LAYERS)
    for name, spec in (doc.get("layers") or {}).items():
        name = str(name).lower()
        spec = dict(spec or {})
        bad = set(spec) - set(LAYER_KEYS)
        if bad:
            raise ValueError(
                f"{source}: layer {name}: unknown key(s) {sorted(bad)} "
                f"(keys: {', '.join(LAYER_KEYS)})"
            )
        base = layers.get(name)
        if base is None and not spec.get("file"):
            raise ValueError(f"{source}: layer {name}: a new layer needs file:")
        change: dict[str, Any] = {}
        for key, value in spec.items():
            if key == "file":
                change["path"] = str(value)
            elif key == "gmt":
                change["gmt"] = dict(value or {})
            elif key in ("pen", "fill", "symbol", "label", "font"):
                change[key] = _none(value)
            else:
                change[key] = str(value)
        if base is None:
            change.setdefault("help", change["path"])
            change.setdefault("build", "(see map_layers.yaml)")
            layers[name] = GisLayer(name, **change)
        else:
            layers[name] = dataclasses.replace(base, **change)
    presets = dict(PRESETS)
    for name, members in (doc.get("presets") or {}).items():
        listed = members.split(",") if isinstance(members, str) else members
        names = tuple(str(m).strip().lower() for m in listed or () if str(m).strip())
        missing = [m for m in names if m not in layers]
        if missing:
            raise ValueError(f"{source}: preset {name}: no layer {', '.join(missing)}")
        presets[str(name).lower()] = names
    return Catalogue(layers, presets, source or "built-in")


def load_catalogue(path: str | None) -> Catalogue:
    """The catalogue of ``path`` (a ``map_layers.yaml``); the built-in one
    when ``path`` is None or not a file.

    Raises:
        ValueError: the file is not a valid catalogue (:func:`catalogue_from`).
    """
    if not path or not os.path.isfile(path):
        return BUILTIN
    import yaml

    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return catalogue_from(doc, path)


def register_steps(cat: Catalogue) -> None:
    """Make each layer of ``cat`` a ``--gmt`` drawing step (a new one of
    the file draws with ``Figure.plot`` like the built-in ones)."""
    from gps_plot import gmt_overrides

    for layer in cat.layers.values():
        gmt_overrides.LAYERS.setdefault(
            layer.step,
            ("Figure.plot", {"style": "S", "pen": "W", "fill": "G"}),
        )


#: ``--layers list`` prints the layers and presets
LIST = "list"


def expand(specs: Sequence[str], cat: Catalogue = BUILTIN) -> list[str]:
    """Layer names of ``--layers`` specs (comma lists, presets expanded,
    repeats kept once, in drawing order as given). ``-NAME`` (a layer or a
    preset) takes out what is in so far: ``volcanic,roads -volcanic`` is
    roads, so an area's map can drop a layer its kind of area gives.

    Raises:
        ValueError: an unknown name, with the known ones.
    """
    out: list[str] = []
    for spec in specs:
        for word in (n.strip().lower() for n in spec.split(",")):
            drop = word.startswith("-")
            name = word.lstrip("-").strip()
            if not name:
                continue
            for n in cat.presets.get(name, (name,)):
                if n not in cat.layers:
                    known = ", ".join([*cat.layers, *cat.presets])
                    raise ValueError(f"--layers {name!r}: not a layer ({known})")
                if drop:
                    out = [m for m in out if m != n]
                elif n not in out:
                    out.append(n)
    return out


def layer_file(layer: GisLayer, gis_dir: str) -> str:
    """Where ``layer``'s file is on this host."""
    return os.path.join(gis_dir, os.path.expanduser(layer.path))


def overlays(
    names: Sequence[str],
    gis_dir: str,
    cat: Catalogue = BUILTIN,
    *,
    strict: bool = True,
    quiet: bool = False,
) -> tuple[Any, ...]:
    """The :class:`gps_plot.maps.MapOverlay` of each layer, in order. With
    ``strict`` False a layer whose file is not on this host is left out,
    with a warning on stderr (``quiet``: without).

    Raises:
        FileNotFoundError: ``strict`` and a layer file not built, naming
            what builds it.
    """
    import sys

    from gps_plot.maps import DEFAULT_PLACE_FONT, MapOverlay

    out = []
    for name in names:
        layer = cat.layers[name]
        path = layer_file(layer, gis_dir)
        if not os.path.isfile(path):
            msg = (
                f"--layers {name}: no {layer.path} in {gis_dir}; built by {layer.build}"
            )
            if strict:
                raise FileNotFoundError(msg)
            if not quiet:
                print(f"{msg}: drawn without it", file=sys.stderr)
            continue
        out.append(
            MapOverlay(
                path, pen=layer.pen, fill=layer.fill, label=layer.label,
                font=layer.font or DEFAULT_PLACE_FONT, symbol=layer.symbol,
                step=layer.step, gmt=dict(layer.gmt),
            )
        )  # fmt: skip
    return tuple(out)


def describe(cat: Catalogue = BUILTIN, gis_dir: str | None = None) -> str:
    """``--layers list``: every layer and preset, and the catalogue's source;
    with ``gis_dir`` the layers whose file is not on this host marked."""
    lines = [f"layers ({cat.source}):"]
    for n, layer in cat.layers.items():
        gone = gis_dir is not None and not os.path.isfile(layer_file(layer, gis_dir))
        lines.append(
            f"  {n:<10} {layer.help}" + (f"  [missing: {layer.build}]" if gone else "")
        )
    lines.append("presets:")
    lines += [f"  {n:<14} {', '.join(m)}" for n, m in cat.presets.items()]
    return "\n".join(lines)


def dump(cat: Catalogue = BUILTIN) -> dict[str, Any]:
    """``cat`` as a ``map_layers.yaml`` document (every key written)."""
    layers = {}
    for name, layer in cat.layers.items():
        entry: dict[str, Any] = {"file": layer.path, "help": layer.help}
        for key in ("pen", "fill", "label", "font", "symbol"):
            if getattr(layer, key) is not None:
                entry[key] = getattr(layer, key)
        entry["build"] = layer.build
        if layer.gmt:
            entry["gmt"] = dict(layer.gmt)
        layers[name] = entry
    return {"layers": layers, "presets": {n: list(m) for n, m in cat.presets.items()}}
