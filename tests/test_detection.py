"""Occupancy, colour and small/big decisions, on drawn-in-code strips.

These use synthetic images so the tests stay fast and never depend on the
user's screenshots: diamonds are convex (solidity 1.0, so "small") and
4-point stars are concave (solidity ~0.5, so "big").
"""

import cv2
import numpy as np
import pytest

import cube_counter as cc

WIDTH = cc.CROP_BOX[2] - cc.CROP_BOX[0]
HEIGHT = cc.CROP_BOX[3] - cc.CROP_BOX[1]
SLOT_W = WIDTH // cc.NUM_SLOTS
GEM_H = int(HEIGHT * cc.GEM_ZONE_BOTTOM_FRAC)

RED = (48, 32, 214)       # BGR; hue ~173 in OpenCV's 0-179 range
GREEN = (90, 200, 30)
BLUE = (230, 120, 20)
YELLOW = (40, 190, 240)


def bgr_for_hue(hue, sat=200, val=220):
    hsv = np.uint8([[[hue, sat, val]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def blank_strip():
    """Dark, unsaturated background: no colour mask and no pill can match it."""
    return np.full((HEIGHT, WIDTH, 3), (105, 105, 110), np.uint8)


def add_pill(img, slot, colour=(238, 238, 240)):
    y0, y1 = int(HEIGHT * cc.PILL_ZONE_FRAC[0]), int(HEIGHT * cc.PILL_ZONE_FRAC[1])
    cv2.rectangle(img, (slot * SLOT_W + 18, y0), ((slot + 1) * SLOT_W - 18, y1), colour, -1)


def add_diamond(img, slot, colour):
    cx, cy, r = slot * SLOT_W + SLOT_W // 2, GEM_H // 2, 58
    points = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], np.int32)
    cv2.fillPoly(img, [points], colour)


def add_star(img, slot, colour, outer=58, inner=20):
    cx, cy = slot * SLOT_W + SLOT_W // 2, GEM_H // 2
    radii = [outer, inner] * 4
    points = np.array([[int(cx + r * np.sin(k * np.pi / 4)),
                        int(cy - r * np.cos(k * np.pi / 4))]
                       for k, r in enumerate(radii)], np.int32)
    cv2.fillPoly(img, [points], colour)


def test_occupancy_follows_the_pills():
    img = blank_strip()
    for slot in (0, 2, 5):
        add_pill(img, slot)
    assert cc.find_occupied_slots(img) == [0, 2, 5]


def test_pill_needs_enough_light_pixels():
    img = blank_strip()
    y0, y1 = int(HEIGHT * cc.PILL_ZONE_FRAC[0]), int(HEIGHT * cc.PILL_ZONE_FRAC[1])
    # A thin sliver of pill: present, but not enough to count as an item.
    cv2.rectangle(img, (20, y0), (SLOT_W - 20, y0 + 3), (238, 238, 240), -1)
    assert cc.find_occupied_slots(img) == []


def test_colours_and_sizes():
    img = blank_strip()
    add_diamond(img, 0, bgr_for_hue(170))     # red small
    add_star(img, 1, bgr_for_hue(80))         # green big
    add_star(img, 2, bgr_for_hue(110))        # blue big
    add_diamond(img, 3, bgr_for_hue(28))      # yellow small

    masks = cc.apply_color_thresholds(img[:GEM_H])
    slots = cc.detect_gems_by_grid(masks, [0, 1, 2, 3])

    assert slots[0]["label"] == "red small"
    assert slots[1]["label"] == "green big"
    assert slots[2]["label"] == "blue big"
    assert slots[3]["label"] == "yellow small"
    for info in slots.values():
        assert info["signal_pct"] >= cc.MIN_SLOT_PCT


def test_solidity_separates_diamond_from_star():
    diamond = blank_strip()
    add_diamond(diamond, 0, bgr_for_hue(170))
    star = blank_strip()
    add_star(star, 0, bgr_for_hue(170))

    solidity = {}
    for name, img in (("diamond", diamond), ("star", star)):
        mask = cc.apply_color_thresholds(img[:GEM_H])["red"]
        _size, solidity[name] = cc.classify_size(cc._slot_slice(mask, 0))

    assert solidity["diamond"] >= cc.SIZE_SOLIDITY_THRESHOLD
    assert solidity["star"] < cc.SIZE_SOLIDITY_THRESHOLD


def test_occupied_slot_without_a_colour_is_reported(capsys):
    img = blank_strip()
    add_pill(img, 0)
    add_pill(img, 1)
    masks = cc.apply_color_thresholds(img[:GEM_H])
    slots = cc.detect_gems_by_grid(masks, [0, 1])
    assert slots == {}
    assert "no colour matched" in capsys.readouterr().out


def test_count_gems_skips_the_orb(tmp_path, monkeypatch):
    img = blank_strip()
    add_diamond(img, 0, bgr_for_hue(170))
    add_star(img, 1, bgr_for_hue(80))
    add_pill(img, 0)
    add_pill(img, 1)
    add_pill(img, 3)                      # the orb: last occupied slot
    path = tmp_path / "strip.png"
    assert cv2.imwrite(str(path), img)

    quantities = {0: 4_100, 1: 325}

    def fake_ocr_all_slots(img_bgr, slots, debug_dir=None, tag=""):
        return {i: {**info, "count": quantities[i]} for i, info in slots.items()}

    monkeypatch.setattr(cc, "ocr_all_slots", fake_ocr_all_slots)
    slots, inventory = cc.count_gems(str(path))

    assert sorted(slots) == [0, 1]        # orb at slot 3 never reaches the inventory
    assert inventory == {"red small": 4_100, "green big": 325}


def test_auto_crop_box_tells_full_screenshots_from_strips(tmp_path):
    strip_path = tmp_path / "strip.png"
    full_path = tmp_path / "full.png"
    assert cv2.imwrite(str(strip_path), blank_strip())
    assert cv2.imwrite(str(full_path), np.zeros((1080, 1920, 3), np.uint8))

    assert cc.auto_crop_box(str(strip_path)) is None
    assert cc.auto_crop_box(str(full_path)) == cc.CROP_BOX
