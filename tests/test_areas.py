"""gps_plot.areas: map file stems by area and selection."""

from gps_plot import areas


def test_slug_folds_icelandic_and_option_dashes():
    assert areas.slug("--epos") == "epos"
    assert areas.slug("subtype = 'SIL stöð'") == "subtype-SIL-stod"
    assert areas.slug("Þeistareykir æ ð") == "Theistareykir-ae-d"
    assert areas.slug("SEY?") == "SEY" and areas.slug("*") == ""


def test_area_is_the_named_box_or_coordinates():
    assert areas.area_name((-24.6, -13.2, 63.2, 66.7)) == "iceland"
    # the overview's widened box is still Iceland
    assert areas.area_name((-25.9, -12.4, 62.6, 67.0)) == "iceland"
    assert areas.area_name((-22.8, -21.9, 63.75, 64.1)) == "reykjanes"
    assert areas.area_name((-14.5, -13.5, 64.8, 65.8)) == "14.5W-13.5W_64.8N-65.8N"


def test_selection_and_stem():
    assert areas.selection(tos=True) == "tos"
    assert areas.selection(tos_search="--epos") == "tos-epos"
    assert areas.selection(tos=True, files=["Planned"]) == "tos+Planned"
    assert areas.selection(patterns=["SEY?", "RHOF"]) == "SEY-RHOF"
    assert areas.selection(patterns=["*"]) == ""
    assert (
        areas.stem("station", (-24.6, -13.2, 63.2, 66.7), "tos-epos")
        == "map_station_iceland_tos-epos"
    )
    assert areas.stem("base", (-22.8, -21.9, 63.75, 64.1), "") == "map_base_reykjanes"
