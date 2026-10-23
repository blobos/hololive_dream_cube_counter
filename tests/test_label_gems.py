"""The typing rules and the answer-key export of label_gems.py."""

import json

import pytest

lg = pytest.importorskip("label_gems")


@pytest.mark.parametrize("raw, expected", [
    ("", 0),
    ("4100", 4_100),
    ("4.1K", 4_100),
    ("1,800", 1_800),
    ("1.2k", 1_200),
    ("4 K", 4_000),
    ("325", 325),
    ("0", 0),
    ("12x", None),
    (".5", None),
    ("abc", None),
])
def test_parse_value(raw, expected):
    assert lg.parse_value(raw) == expected


def test_natural_sort_puts_2_before_10():
    names = ["Screenshot (10).png", "Screenshot (2).png", "a.png"]
    assert sorted(names, key=lg.natural_key) == ["a.png", "Screenshot (2).png",
                                                 "Screenshot (10).png"]


def state_with(images):
    return {"version": 1, "images": images}


def test_export_separates_reviewed_flagged_skipped_unreviewed(tmp_path, capsys):
    state = state_with({
        "a.png": {"values": {"red small": 4_100}, "reviewed": True},
        "b.png": {"values": {"red small": 100, "blue big": 7}, "reviewed": True,
                  "flag": True},
        "c.png": {"skipped": True},
        "d.png": {"values": {}, "reviewed": False},
    })
    out = tmp_path / "expected.json"
    payload = lg.export_labels(state, out)
    written = json.loads(out.read_text())

    assert written == payload
    assert written["source_images"] == 2
    assert written["per_image"] == {"a.png": {"red small": 4_100},
                                    "b.png": {"red small": 100, "blue big": 7}}
    assert written["totals"] == {"red small": 4_200, "blue big": 7}
    assert list(written["totals"]) == ["red small", "blue big"]   # LABEL_ORDER
    assert written["flagged"] == ["b.png"]
    assert written["skipped"] == ["c.png"]
    assert written["unreviewed"] == ["d.png"]
    assert "Verified totals:" in capsys.readouterr().out


def test_export_points_at_values_the_pipeline_got_wrong(tmp_path, capsys):
    state = state_with({"a.png": {"values": {"red small": 4_100}, "reviewed": True}})
    prefill = {"images": {"a.png": {"red small": 4_000, "blue big": 7}}}
    lg.export_labels(state, tmp_path / "expected.json", prefill=prefill)

    printed = capsys.readouterr().out
    assert "pipeline" in printed
    assert "4,000" in printed and "4,100" in printed
    assert "6/8 values agree" in printed      # the two differing values above
