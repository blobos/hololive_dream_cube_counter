"""Reading what the game prints inside the quantity pill."""

import pytest

import cube_counter as cc


@pytest.mark.parametrize("raw, expected", [
    ("", 0),
    (None, 0),
    ("40", 40),
    ("325", 325),
    ("820", 820),
    ("1K", 1_000),
    ("4.1K", 4_100),
    ("21.5K", 21_500),
    ("2.5M", 2_500_000),
    ("1.2B", 1_200_000_000),
    ("4.1k", 4_100),          # the reader lowercases sometimes
    ("4.1K.", 4_100),         # stray trailing dot from the detector
    (" 820 ", 820),
    ("325\n", 325),
    ("0", 0),
    ("garbage", 0),
    ("K", 0),
])
def test_parse_quantity(raw, expected):
    assert cc.parse_quantity(raw) == expected


def test_parse_quantity_rounds_to_nearest():
    assert cc.parse_quantity("1.15K") == 1_150


def test_build_inventory_sums_duplicate_labels():
    slots = {
        0: {"label": "red small", "count": 100},
        1: {"label": "red small", "count": 25},
        2: {"label": "blue big", "count": 7},
    }
    returned, inventory = cc.build_inventory(slots)
    assert returned is slots
    assert inventory == {"red small": 125, "blue big": 7}


def test_apply_extra_gems_accepts_numbers_and_strings():
    totals = cc.apply_extra_gems({"red small": 10}, {"red small": "2.2k",
                                                      "blue big": 238,
                                                      "green big": ""})
    assert totals == {"red small": 2_210, "blue big": 238}
