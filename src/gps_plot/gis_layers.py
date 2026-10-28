"""Named map layers: ``gpsplot map --layers lavas,barriers,…``.

A layer is a file under the local GIS directory (``gis_dir`` in
postprocess.cfg, default ``~/gps-data/gis``) with its own look and its own
``--gmt`` drawing step, so ``--gmt roads:pen=0.8p,black`` restyles the
roads alone. ``--overlay FILE`` stays for anything ad hoc.

The Reykjanes layers are built by ``vector/build_08_reykjanes.sh`` (lavas:
the LMÍ/NÍ WFS, saved beside it; places: authored); the national IS 50V
roads, buildings and farm names by ``vector/build_09_is50v.sh``; the
volcanic systems and caldera rims (FutureVolc) by ``build_07_volcanic.sh``. A preset names
several layers in drawing order, bottom first: roads under the lavas that
cut them, the barriers over the lavas they hold back, names last.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
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

#: ``--layers list`` prints the layers and presets
LIST = "list"


def expand(specs: Sequence[str]) -> list[str]:
    """Layer names of ``--layers`` specs (comma lists, presets expanded,
    repeats kept once, in drawing order as given).

    Raises:
        ValueError: an unknown name, with the known ones.
    """
    out: list[str] = []
    for spec in specs:
        for name in (n.strip().lower() for n in spec.split(",")):
            if not name:
                continue
            names = PRESETS.get(name, (name,))
            for n in names:
                if n not in LAYERS:
                    known = ", ".join([*LAYERS, *PRESETS])
                    raise ValueError(f"--layers {name!r}: not a layer ({known})")
                if n not in out:
                    out.append(n)
    return out


def overlays(names: Sequence[str], gis_dir: str) -> tuple[Any, ...]:
    """The :class:`gps_plot.maps.MapOverlay` of each layer, in order.

    Raises:
        FileNotFoundError: a layer file not built, naming what builds it.
    """
    from gps_plot.maps import DEFAULT_PLACE_FONT, MapOverlay

    out = []
    for name in names:
        layer = LAYERS[name]
        path = os.path.join(gis_dir, layer.path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"--layers {name}: no {layer.path} in {gis_dir}; built by {layer.build}"
            )
        out.append(
            MapOverlay(
                path, pen=layer.pen, fill=layer.fill, label=layer.label,
                font=layer.font or DEFAULT_PLACE_FONT, symbol=layer.symbol,
                step="place-labels" if path.endswith(".txt") else name,
            )
        )  # fmt: skip
    return tuple(out)


def describe() -> str:
    """``--layers list``: every layer and preset."""
    lines = ["layers:"]
    lines += [f"  {n:<10} {layer.help}" for n, layer in LAYERS.items()]
    lines.append("presets:")
    lines += [f"  {n:<14} {', '.join(v)}" for n, v in PRESETS.items()]
    return "\n".join(lines)
