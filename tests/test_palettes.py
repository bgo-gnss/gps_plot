"""gps_plot.palettes: colour-map lookup and gap-free stitching."""

from __future__ import annotations

import pytest

from gps_plot import palettes
from gps_plot.palettes import Slice


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    gmt = tmp_path / "gmt"
    (gmt / "SCM").mkdir(parents=True)
    (gmt / "gmt").mkdir()
    (gmt / "gmt" / "geo.cpt").write_text(
        "# HINGE = SOFT\n-1 blue 0 cyan\n0 green 1 white\n"
    )
    (gmt / "gmt" / "gray.cpt").write_text("0 black 1 white\n")
    (gmt / "SCM" / "batlow.cpt").write_text("0 1/25/89 1 250/204/250\n")
    user = tmp_path / "gis" / "cpt"
    user.mkdir(parents=True)
    return gmt, user


def test_resolve_order(dirs, tmp_path):
    gmt, user = dirs
    assert palettes.resolve("slate", user_dir=user, gmt_dir=gmt).source == str(
        palettes.GPSPLOT_DIR / "slate.cpt"
    )
    (user / "slate.cpt").write_text("0 red 1 red\n")  # yours win over gpsplot's
    assert palettes.resolve("Slate", user_dir=user, gmt_dir=gmt).source == str(
        user / "slate.cpt"
    )
    assert palettes.resolve("batlow", user_dir=user, gmt_dir=gmt).kind == palettes.GMT
    assert palettes.resolve("geo+h0", user_dir=user, gmt_dir=gmt).kind == palettes.GMT
    own = tmp_path / "own.cpt"
    own.write_text("0 red 1 red\n")
    assert palettes.resolve(str(own), user_dir=user, gmt_dir=gmt).kind == palettes.FILE
    assert palettes.resolve("white", user_dir=user, gmt_dir=gmt).kind == palettes.COLOUR


def test_hinged_and_listing(dirs):
    gmt, user = dirs
    assert palettes.is_hinged(
        palettes.resolve("geo", user_dir=user, gmt_dir=gmt), gmt_dir=gmt
    )
    assert not palettes.is_hinged(
        palettes.resolve("gray", user_dir=user, gmt_dir=gmt), gmt_dir=gmt
    )
    assert palettes.is_hinged(
        palettes.resolve("sea", user_dir=user, gmt_dir=gmt), gmt_dir=gmt
    )
    assert not palettes.is_hinged(
        palettes.resolve("slate", user_dir=user, gmt_dir=gmt), gmt_dir=gmt
    )
    (user / "mine.cpt").write_text("0 red 1 red\n")
    listing = palettes.available(user_dir=user, gmt_dir=gmt)
    assert listing[f"gpsplot ({palettes.GPSPLOT_DIR})"] == [
        "lightgrey", "relief", "sea", "slate",
    ]  # fmt: skip
    assert listing[f"yours ({user})"] == ["mine"]
    assert listing["GMT SCM"] == ["batlow"] and listing["GMT gmt"] == ["geo", "gray"]
    assert palettes.suggest("slade", user_dir=user, gmt_dir=gmt)[0] == "slate"


def test_parse_both_layouts():
    text = "# c\n-200 175 208 232 0 205 228 242\n0 a 2109.0035 b L\nB black\nF white\n"
    assert palettes.parse(text) == [
        Slice(-200.0, "175/208/232", 0.0, "205/228/242"),
        Slice(0.0, "a", 2109.0035, "b"),
    ]


def test_stitch_fills_gaps_and_keeps_full_precision():
    sea = [Slice(-300.0, "blue", 0.0, "cyan")]
    land = [Slice(0.0, "green", 2109.00351562, "grey")]
    ice = palettes.shift([Slice(1500.0, "white", 2000.0, "white")], 100000.0)
    text = palettes.stitch([sea, land, ice])
    palettes.check_gaps(text)
    filler = palettes.parse(text)[2]
    assert (filler.lo, filler.hi, filler.lo_colour) == (
        2109.00351562,
        101500.0,
        "white",
    )
    with pytest.raises(ValueError, match="overlap"):
        palettes.stitch([land, sea])
    with pytest.raises(ValueError, match="do not join"):
        palettes.check_gaps("0 a 2109.0035 b\n2109 white 3000 white\n")
