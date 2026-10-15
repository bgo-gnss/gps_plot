"""gps_plot.gmt_overrides: --gmt STEP:KEY=VALUE on the map drawing steps."""

from __future__ import annotations

import pytest

from gps_plot import gmt_overrides as go


def test_parse_specs():
    assert go.parse(
        ["coast:A=500/0/1", "glaciers:pen=0.3p,blue", "coast:Q", "coast:A=100"]
    ) == {  # fmt: skip
        "coast": {"A": "100", "Q": True},
        "glaciers": {"pen": "0.3p,blue"},
    }
    for bad in ("coast", "coast:", ":A=1", "coast:=1"):
        with pytest.raises(ValueError, match="LAYER:KEY=VALUE"):
            go.parse([bad])


def test_layer_kw_replaces_by_long_name_or_letter():
    assert go.layer_kw("glaciers", pen="a", fill="b") == {"pen": "a", "fill": "b"}
    with go.overrides({"glaciers": {"W": "0.3p,blue"}, "lakes": {"fill": "red"}}):
        assert go.layer_kw("glaciers", data="g", pen="a", fill="b") == {
            "data": "g", "fill": "b", "W": "0.3p,blue",
        }  # fmt: skip
        assert go.layer_kw("lakes", pen="a", fill="b") == {"pen": "a", "fill": "red"}
        assert go.layer_kw("coastline", pen="a") == {"pen": "a"}
    # text folds font and justify into -F: one F drops both
    with go.overrides({"labels": {"F": "+f6p,red+jTL"}}):
        assert go.layer_kw("labels", font="8p", justify="BL", fill="white") == {
            "fill": "white", "F": "+f6p,red+jTL",
        }  # fmt: skip
    # the shoreline-only coast call takes no fill
    with go.overrides({"coast": {"land": "tan", "A": "500"}}):
        assert go.layer_kw("coast", exclude=("land", "G"), shorelines=True) == {
            "shorelines": True, "A": "500",
        }  # fmt: skip


def test_overrides_reset_after_an_error():
    with pytest.raises(RuntimeError), go.overrides({"lakes": {"fill": "red"}}):
        raise RuntimeError
    assert go.layer_kw("lakes", fill="b") == {"fill": "b"}


def test_check_refuses_unknown_and_owned_keys():
    pygmt = pytest.importorskip("pygmt")
    go.check({"coast": {"A": "500", "area_thresh": "500"}, "dem": {"t": "20"}}, pygmt)
    with pytest.raises(ValueError, match="did you mean glaciers"):
        go.check({"glacier": {"pen": "1p"}}, pygmt)
    with pytest.raises(ValueError, match="did you mean.*pen"):
        go.check({"glaciers": {"pne": "1p"}}, pygmt)
    with pytest.raises(ValueError, match="--dem-cmap"):
        go.check({"dem": {"C": "geo"}}, pygmt)
    with pytest.raises(ValueError, match="sets 'region' itself"):
        go.check({"coast": {"region": "0/1/0/1"}}, pygmt)


#: each (long name, letter) of LAYERS, with what else the call needs to get
#: as far as PyGMT's parameter check; text's F is the exception (no error)
def _pairs():
    for layer, (call, letters) in go.LAYERS.items():
        for long, letter in letters.items():
            if call == "Figure.text":
                continue
            yield layer, call, long, letter


@pytest.mark.parametrize(("layer", "call", "long", "letter"), list(_pairs()))
def test_alias_table_pairs_conflict_in_pygmt(layer, call, long, letter, tmp_path):
    pygmt = pytest.importorskip("pygmt")
    from pygmt.exceptions import GMTParameterError

    extra: dict = {}
    if call == "Figure.coast" and long in ("resolution", "frame"):
        extra = {"shorelines": True}
    if call == "grdgradient" and long == "normalize":
        extra = {"azimuth": 315}
    if call == "grdfilter":  # both required, checked before the conflict
        extra = {"filter": "g3", "distance": 2}
        extra.pop(long)
    if call == "Figure.image":
        extra = {"imagefile": str(tmp_path / "x.png")}
    fig = pygmt.Figure()
    fig.basemap(region=[0, 1, 0, 1], projection="X2c", frame=True)
    target = (
        getattr(pygmt, call) if "." not in call else getattr(fig, call.split(".")[1])
    )
    if "." not in call:  # grdgradient, grdfilter
        import numpy as np
        import xarray as xr

        extra["grid"] = xr.DataArray(
            np.zeros((3, 3)),
            coords={"y": [0.0, 1, 2], "x": [0.0, 1, 2]},
            dims=("y", "x"),
        )
    # values PyGMT validates itself
    value = {"resolution": "low", "filter": "g3", "distance": 2}.get(long, "x")
    with pytest.raises(GMTParameterError, match="(?i)conflict|exclusive"):
        target(**extra, **{long: value, letter: "y"})
