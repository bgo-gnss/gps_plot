"""Regional maps: named GIS layers, place names, decluttered station names,
insets (Iceland locator, zoom box) and their placement.

Pure tests run anywhere; the render test needs PyGMT and a GMT library."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from gps_plot import declutter, gis_layers, insets, layout, maps
from gps_plot import map as gmap

# ---------------------------------------------------------------------------
# declutter
# ---------------------------------------------------------------------------


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _boxes(xy, texts, labels, size=5.0):
    out = []
    for lb in labels:
        w, h = declutter.text_size(texts[lb.index], size)
        out.append(declutter._box(lb.x, lb.y, lb.justify, w, h))
    return out


def test_open_points_keep_the_default_spot_up_and_right():
    xy = np.array([[2.0, 2.0], [8.0, 5.0]])
    labels = declutter.place(xy, ["AAAA", "BBBB"], size_pt=5, width=12, height=8)
    assert [lb.justify for lb in labels] == ["BL", "BL"]
    assert not any(lb.leader for lb in labels)
    assert all(lb.x > p[0] and lb.y > p[1] for lb, p in zip(labels, xy, strict=True))


def test_a_cluster_gets_names_that_do_not_overlap_each_other_or_symbols():
    rng = np.random.default_rng(1)
    xy = 5.0 + rng.uniform(-0.6, 0.6, size=(15, 2))
    texts = [f"S{k:03d}" for k in range(15)]
    labels = declutter.place(xy, texts, size_pt=5, width=12, height=10)
    boxes = _boxes(xy, texts, labels)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not _overlap(boxes[i], boxes[j]), (texts[i], texts[j])
        for k, (x, y) in enumerate(xy):
            if k != labels[i].index:
                sym = (x - declutter.SYMBOL_CM, x + declutter.SYMBOL_CM,
                       y - declutter.SYMBOL_CM, y + declutter.SYMBOL_CM)  # fmt: skip
                assert not _overlap(boxes[i], sym)
    assert any(lb.leader for lb in labels)  # too dense for the near spots alone


def test_names_keep_inside_the_map_and_off_fixed_boxes():
    xy = np.array([[11.9, 7.9]])  # the top-right corner: up-right is outside
    fixed = [declutter.centred_box(11.5, 7.5, "Svartsengi", 6)]
    (lb,) = declutter.place(xy, ["SENG"], size_pt=5, width=12, height=8, fixed=fixed)
    box = _boxes(xy, ["SENG"], [lb])[0]
    assert box[0] >= 0 and box[1] <= 12 and box[2] >= 0 and box[3] <= 8
    assert not _overlap(box, fixed[0])


def test_no_free_spot_falls_back_to_the_default():
    xy = np.array([[0.5, 0.5]])
    (lb,) = declutter.place(xy, ["TOOLONGNAME"], size_pt=5, width=1.0, height=1.0)
    assert lb.justify == "BL" and not lb.leader


# ---------------------------------------------------------------------------
# GIS layers, place names, overlay labels
# ---------------------------------------------------------------------------


def test_layers_expand_presets_in_drawing_order_once():
    assert gis_layers.expand(["eruptions", "lavas,barriers"]) == [
        "lavas", "graben", "fissures", "barriers",
    ]  # fmt: skip
    assert gis_layers.expand(["reykjanes"])[-1] == "places"
    with pytest.raises(ValueError, match="'nope': not a layer"):
        gis_layers.expand(["nope"])


def test_layers_name_their_files_and_steps(tmp_path):
    for name in ("roads", "places"):
        path = tmp_path / gis_layers.LAYERS[name].path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    roads, places = gis_layers.overlays(["roads", "places"], str(tmp_path))
    assert roads.step == "roads" and roads.pen == gis_layers.LAYERS["roads"].pen
    assert places.step == "place-labels"
    with pytest.raises(FileNotFoundError, match="build_08_reykjanes.sh"):
        gis_layers.overlays(["barriers"], str(tmp_path))


def test_place_names_read_lon_lat_and_the_rest_of_the_line(tmp_path):
    f = tmp_path / "places.txt"
    f.write_text("# comment\n\n-22.449 63.8802 Bláa lónið\n-22.43 63.88 Svartsengi\n")
    assert maps.place_names(f) == [
        (-22.449, 63.8802, "Bláa lónið"), (-22.43, 63.88, "Svartsengi"),
    ]  # fmt: skip
    f.write_text("-22.4 Grindavík\n")
    with pytest.raises(ValueError, match=":1: not 'LON LAT NAME'"):
        maps.place_names(f)


def test_overlay_labels_sit_in_polygons_and_on_lines():
    gj = {"features": [
        {"properties": {"name": "Box"}, "geometry": {"type": "Polygon",
         "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2]]]}},
        {"properties": {"name": "Line"}, "geometry": {"type": "LineString",
         "coordinates": [[0, 0], [1, 1], [2, 2]]}},
        {"properties": {"name": ""}, "geometry": {"type": "Point", "coordinates": [0, 0]}},
    ]}  # fmt: skip
    assert maps.overlay_labels(gj, "name") == [(1.0, 1.0, "Box"), (1.0, 1.0, "Line")]


def test_overlay_label_points_leave_out_the_inset_box(tmp_path):
    f = tmp_path / "places.txt"
    f.write_text("-22.43 63.88 Svartsengi\n-22.07 63.87 Krýsuvík\n")
    ov = maps.MapOverlay(str(f), unlabelled_box=(-22.6, -22.3, 63.8, 63.95))
    assert [p[2] for p in maps.overlay_label_points([ov])] == ["Krýsuvík"]


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------


def _sea(width=12.0, height=8.0, land=()):
    ny, nx = round(height / layout.CELL_CM), round(width / layout.CELL_CM)
    sea = np.ones((ny, nx), dtype=bool)
    for x0, x1, y0, y1 in land:
        sea[round(y0 / layout.CELL_CM) : round(y1 / layout.CELL_CM),
            round(x0 / layout.CELL_CM) : round(x1 / layout.CELL_CM)] = False  # fmt: skip
    return layout.SeaLayout(width, height, sea)


def test_an_inset_takes_land_when_no_sea_is_free_but_never_a_symbol():
    lay = _sea(land=[(0.0, 12.0, 0.0, 8.0)])
    assert lay.place((3.0, 2.0), "BR") is None
    mask = np.zeros_like(lay.sea)
    mask[:, 60:] = True  # symbols over the right half
    lay.block(mask)
    x, y = lay.place((3.0, 2.0), "BR", on_land=True, reach=12.0)
    assert x + 3.0 <= 6.0 + 1e-9


def test_place_falls_back_beside_the_map_and_stacks_below():
    lay = _sea()
    lay.block(np.ones_like(lay.sea))
    spot = insets.place(lay, (6.0, 4.0), ("BR",), outside=True, map_size=(12.0, 8.0))
    assert spot.beside and spot.x == pytest.approx(12.0 + insets.BESIDE_GAP_CM)
    assert spot.y == pytest.approx(4.0)
    below = insets.beside((3.0, 2.0), (12.0, 8.0), below=spot)
    assert below.x == spot.x and below.y == pytest.approx(
        4.0 - insets.BESIDE_GAP_CM - 2.0
    )
    assert insets.place(lay, (6.0, 4.0), ("BR",)) is None


def test_locator_default_on_for_regions_off_for_all_iceland():
    args = argparse.Namespace(locator=None)
    assert gmap._wants_locator(args, (-22.9, -21.7, 63.75, 64.1))
    assert not gmap._wants_locator(args, (-24.6, -13.2, 63.2, 66.7))
    assert not gmap._wants_locator(
        argparse.Namespace(locator=False), (-22.9, -21.7, 63.75, 64.1)
    )


def test_region_takes_an_area_name():
    from gps_plot import areas

    assert gmap._region("Svartsengi") == areas.AREAS["svartsengi"]
    with pytest.raises(argparse.ArgumentTypeError, match="area name"):
        gmap._region("atlantis")


# ---------------------------------------------------------------------------
# render (PyGMT + GMT)
# ---------------------------------------------------------------------------


def test_station_map_with_layers_and_both_insets_renders(tmp_path, monkeypatch):
    pytest.importorskip("pygmt")
    try:
        maps._mapproject("-R0/1/0/1 -JM5c -W")
    except Exception:  # noqa: BLE001 - no GMT library
        pytest.skip("GMT library not available")
    gis = tmp_path / "gis"
    (gis / "vector" / "reykjanes").mkdir(parents=True)
    (gis / "vector" / "reykjanes" / "places.txt").write_text(
        "-22.43 63.88 Svartsengi\n"
    )
    (gis / "vector" / "reykjanes" / "barriers.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
         "geometry": {"type": "LineString",
                      "coordinates": [[-22.45, 63.87], [-22.42, 63.88]]}}]}))  # fmt: skip
    monkeypatch.setattr(gmap, "gis_dir", lambda config: str(gis))
    pts = [maps.StationCoordinate(f"S{k:03d}", -22.45 + 0.01 * (k % 5),
                                  63.86 + 0.008 * (k // 5)) for k in range(15)]  # fmt: skip
    pts.append(maps.StationCoordinate("FARR", -22.0, 64.0))

    class Config:
        def getStationInfo(self):
            return [p.marker for p in pts]

    monkeypatch.setattr(maps, "station_coordinates",
                        lambda markers, parser=None: [p for p in pts if p.marker in markers])  # fmt: skip
    p = gmap.build_parser(config=None)
    args = p.parse_args(["station", "*", "--region", "reykjanes", "--layers",
                         "barriers,places", "--inset", "svartsengi", "--save", "png",
                         "-d", str(tmp_path), "--name", "regional"])  # fmt: skip
    assert gmap.run(args, p, Config()) == 0
    assert (tmp_path / "regional.png").stat().st_size > 10_000


def test_a_station_with_several_vectors_is_named_once():
    calls = []

    class Fig:
        def text(self, **kw):
            calls.append(kw)

    pts = [maps.StationCoordinate("SENG", -22.4, 63.9),
           maps.StationCoordinate("SENG", -22.4, 63.9),
           maps.StationCoordinate("SKSH", -22.5, 63.9)]  # fmt: skip
    maps._draw_station_labels(Fig(), pts, set(), "5p", "J0.2")
    assert calls[0]["text"] == ["SENG", "SKSH"]


def test_area_layers_volcanic_and_rural(tmp_path):
    from gps_plot import gmt_overrides

    assert gis_layers.expand(["volcanic,rural"]) == [
        "swarms", "volcanoes", "calderas", "localroads", "roads", "buildings", "farms",
    ]  # fmt: skip
    assert set(gis_layers.expand(["volcanic,rural"])) - {"farms"} <= set(
        gmt_overrides.LAYERS
    )
    for name in ("buildings", "farms"):
        path = tmp_path / gis_layers.LAYERS[name].path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    buildings, farms = gis_layers.overlays(["buildings", "farms"], str(tmp_path))
    assert buildings.symbol == "s0.07c" and buildings.step == "buildings"
    assert farms.step == "place-labels"
    with pytest.raises(FileNotFoundError, match="build_09_is50v.sh"):
        gis_layers.overlays(["localroads"], str(tmp_path))


def test_overlay_points_and_labels_kept_apart():
    from gps_plot import maps

    doc = {"features": [
        {"geometry": {"type": "Point", "coordinates": [-16.9, 64.0]}},
        {"geometry": {"type": "MultiPoint", "coordinates": [[-16.8, 64.0], [-16.7, 64.0]]}},
        {"geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}},
    ]}  # fmt: skip
    assert maps.overlay_points(doc) == [(-16.9, 64.0), (-16.8, 64.0), (-16.7, 64.0)]
    region, projection = (-17.0, -16.6, 63.9, 64.05), "M10c"
    labels = [(-16.85, 63.99, "Svínafell"), (-16.851, 63.991, "Nýjahús"),
              (-16.7, 63.95, "Hof")]  # fmt: skip
    kept = maps.apart(labels, "6p,Helvetica", region, projection)
    assert [t for *_, t in kept] == ["Svínafell", "Hof"]  # the first one wins


def test_catalogue_file_adds_and_changes_layers(tmp_path):
    from gps_plot import gmt_overrides

    doc = {
        "layers": {
            "calderas": {"pen": "2p,red"},  # a built-in one, restyled
            "huts": {"file": "vector/huts.geojson", "symbol": "t0.1c",
                     "fill": "orange", "pen": "none", "gmt": {"transparency": 30}},
        },
        "presets": {"oraefi": ["volcanoes", "calderas", "huts"]},
    }  # fmt: skip
    cat = gis_layers.catalogue_from(doc, "test.yaml")
    assert cat.layers["calderas"].pen == "2p,red"
    assert cat.layers["calderas"].path == gis_layers.LAYERS["calderas"].path
    huts = cat.layers["huts"]
    assert (huts.pen, huts.symbol, huts.step) == (None, "t0.1c", "huts")
    assert gis_layers.expand(["oraefi"], cat) == ["volcanoes", "calderas", "huts"]
    assert gis_layers.LAYERS["calderas"].pen != "2p,red"  # the built-in kept
    (tmp_path / "vector").mkdir()
    (tmp_path / "vector/huts.geojson").write_text("{}")
    (ov,) = gis_layers.overlays(["huts"], str(tmp_path), cat)
    assert ov.gmt == {"transparency": 30} and ov.step == "huts"
    gis_layers.register_steps(cat)
    assert "huts" in gmt_overrides.LAYERS  # --gmt huts:pen=… is valid
    # the dump reads back as the same catalogue
    again = gis_layers.catalogue_from(gis_layers.dump(cat), "dump")
    assert again.layers == cat.layers and again.presets == cat.presets


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ({"layers": {"x": {"pen": "1p"}}}, "needs file"),
        ({"layers": {"roads": {"colour": "red"}}}, "unknown key"),
        ({"presets": {"p": ["nope"]}}, "no layer nope"),
        ({"styles": {}}, "unknown key"),
    ],
)
def test_catalogue_file_refuses(doc, match):
    with pytest.raises(ValueError, match=match):
        gis_layers.catalogue_from(doc, "t.yaml")


def test_load_catalogue_falls_back_to_the_builtin(tmp_path):
    assert gis_layers.load_catalogue(None) is gis_layers.BUILTIN
    assert gis_layers.load_catalogue(str(tmp_path / "none.yaml")) is gis_layers.BUILTIN
    f = tmp_path / "map_layers.yaml"
    f.write_text("presets:\n  mine: roads, farms\n")
    cat = gis_layers.load_catalogue(str(f))
    assert cat.presets["mine"] == ("roads", "farms") and cat.source == str(f)


def test_a_layers_own_gmt_is_replaced_by_the_users_under_either_name():
    from gps_plot import gmt_overrides

    cat = gis_layers.catalogue_from(
        {"layers": {"huts": {"file": "h.geojson", "gmt": {"transparency": 20}}}}
    )
    gis_layers.register_steps(cat)
    with gmt_overrides.overrides({"huts": {"t": "50"}}):
        kw = gmt_overrides.layer_kw("huts", data="x", transparency=20)
    assert kw == {"data": "x", "t": "50"}


def test_missing_layer_files_are_skipped_and_listed(tmp_path, capsys):
    (tmp_path / "vector/is50v").mkdir(parents=True)
    (tmp_path / "vector/is50v/roads.geojson").write_text("{}")
    (roads,) = gis_layers.overlays(["roads", "buildings"], str(tmp_path), strict=False)
    assert roads.step == "roads"
    assert "buildings" in capsys.readouterr().err  # said, not fatal
    listing = gis_layers.describe(gis_layers.BUILTIN, str(tmp_path))
    line = next(x for x in listing.splitlines() if x.strip().startswith("buildings"))
    assert "[missing: vector/build_09_is50v.sh]" in line
    assert "missing" not in next(
        x for x in listing.splitlines() if x.strip().startswith("roads ")
    )


def test_earth_relief_fallback_fits_the_pixels_and_skips_srtm_in_the_north():
    from gps_plot import maps

    # Svínafellsheiði at 15.5 cm: finer than 15″ pixels, but no SRTM up there
    assert maps.earth_relief_fallback((-17.06, -16.58, 63.9, 64.05), "M15.5c") == (
        "@earth_relief_15s"
    )
    # the same small map at 45°N takes SRTM's finest (~1″ pixels)
    assert maps.earth_relief_fallback((-17.06, -16.58, 45.0, 45.15), "M15.5c") == (
        "@earth_relief_01s"
    )
    # all of Iceland (~24″ pixels): 15″
    assert maps.earth_relief_fallback((-25.0, -13.0, 63.0, 67.0), "M15c") == (
        "@earth_relief_15s"
    )
    # a 2° pixel world map: 1′
    assert maps.earth_relief_fallback((-180.0, 180.0, -60.0, 60.0), "M3c") == (
        "@earth_relief_01m"
    )
