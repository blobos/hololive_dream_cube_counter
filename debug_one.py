#!/usr/bin/env python3
"""
Debug one screenshot through every stage of the counter
=======================================================
Writes one PNG per stage plus a single contact sheet, and prints the per-slot
table the pipeline decided from. Use it when a number is wrong to tell at a
glance whether the crop, the colour masks, the size call, or the OCR is at
fault.

Usage:
    python debug_one.py ~/Downloads/holocube/"Screenshot (180).png"
    python debug_one.py ~/Downloads/holocube            # every image in a folder
    python debug_one.py shot.png --no-ocr               # skip EasyOCR (fast)
    python debug_one.py strip.png --no-crop --zoom 1.4

Output, in debug_one/<image name>/:
    01_strip.png        crop + slot grid + zone boundaries
    02_masks.png        the four colour masks
    03_occupancy.png    colour % and pill-light % per slot, heat coloured
    04_tiles.png        detected tiles with the contour used for solidity
    05_ocr_crops.png    exactly what EasyOCR is given, per slot
    06_ocr.png          raw OCR text -> parsed quantity
    07_contact_sheet.png  everything above, stacked
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import cube_counter as cc


FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)

BAR_BG = (34, 37, 49)
BAR_FG = (240, 240, 245)
SLOT_LINE = (120, 120, 135)
GEM_LINE = (60, 200, 120)
PILL_LINES = (255, 170, 60)


def font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def to_pil(img):
    """cv2 BGR (or RGB processed crop) -> PIL."""
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def bar(text, width, height=26, bg=BAR_BG, fg=BAR_FG, size=15):
    img = Image.new("RGB", (max(width, 8), height), bg)
    ImageDraw.Draw(img).text((8, max(2, (height - size) // 2 - 2)), text,
                             font=font(size), fill=fg)
    return img


def stack(parts, gap=6, bg=(246, 246, 250)):
    width = max(p.width for p in parts)
    height = sum(p.height for p in parts) + gap * (len(parts) - 1)
    out = Image.new("RGB", (width, height), bg)
    y = 0
    for part in parts:
        out.paste(part, (0, y))
        y += part.height + gap
    return out


def border(img, width=1, colour=(120, 120, 135)):
    out = img.copy()
    ImageDraw.Draw(out).rectangle([0, 0, out.width - 1, out.height - 1], outline=colour,
                                  width=width)
    return out


def tag(draw, xy, text, colour, f):
    """Text on a dark box, so annotations stay readable over any screenshot."""
    box = draw.textbbox(xy, text, font=f)
    draw.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(24, 24, 32))
    draw.text(xy, text, font=f, fill=colour)


def save(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, Image.Image):
        img.save(path)
    else:
        cv2.imwrite(str(path), img)
    return path

# =============================================================================
# THE DETECTION HALF, LAID OUT FOR EYES
# =============================================================================

def load_strip(path, crop=True):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    if not crop:
        return img
    l, t, r, b = cc.CROP_BOX
    if img.shape[1] == r - l and img.shape[0] == b - t:
        return img
    return img[t:b, l:r]


def pill_percentages(img_bgr):
    """Same measurement find_occupied_slots() uses, but returned per slot."""
    h, w, _ = img_bgr.shape
    slot_w = w // cc.NUM_SLOTS
    y0, y1 = int(h * cc.PILL_ZONE_FRAC[0]), int(h * cc.PILL_ZONE_FRAC[1])
    pad_x = int(slot_w * 0.18)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    out = []
    for i in range(cc.NUM_SLOTS):
        zone = hsv[y0:y1, i * slot_w + pad_x:(i + 1) * slot_w - pad_x]
        light = (zone[..., 1] < 50) & (zone[..., 2] > 190)
        out.append(round(float(light.mean()) * 100, 1))
    return out


def scan_image(path, crop=True):
    """Runs the detection half of count_gems() and keeps every intermediate."""
    strip = load_strip(path, crop)
    h = strip.shape[0]
    occupied = cc.find_occupied_slots(strip)
    orb = occupied[-1] if (cc.ORB_IS_LAST_ITEM and occupied) else None
    gem_indices = occupied[:-1] if orb is not None else list(occupied)
    gem_zone = strip[:int(h * cc.GEM_ZONE_BOTTOM_FRAC)]
    masks = cc.apply_color_thresholds(gem_zone)
    slots = cc.detect_gems_by_grid(masks, gem_indices)
    return {"strip": strip, "occupied": occupied, "orb": orb,
            "gem_indices": gem_indices, "masks": masks, "slots": slots,
            "pill_pcts": pill_percentages(strip)}


def stage_strip(scan, zoom=1.0):
    strip = scan["strip"]
    h, w = strip.shape[:2]
    img = strip.copy()
    if zoom != 1.0:
        img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]
    slot_w = int(w / cc.NUM_SLOTS * zoom)

    for i in range(1, cc.NUM_SLOTS):
        cv2.line(img, (i * slot_w, 0), (i * slot_w, H - 1), SLOT_LINE, 1)

    y_gem = int(h * cc.GEM_ZONE_BOTTOM_FRAC * zoom)
    cv2.line(img, (0, y_gem), (W - 1, y_gem), GEM_LINE, 2)
    for frac in cc.PILL_ZONE_FRAC:
        y = int(h * frac * zoom)
        cv2.line(img, (0, y), (W - 1, y), PILL_LINES, 1)

    pil = to_pil(img)
    draw = ImageDraw.Draw(pil)
    small = font(12)
    tag(draw, (8, max(2, y_gem - 18)), "GEM_ZONE_BOTTOM_FRAC", GEM_LINE, small)
    tag(draw, (W - 118, int(h * cc.PILL_ZONE_FRAC[0] * zoom) + 2), "PILL_ZONE_FRAC",
        PILL_LINES, small)

    for i in range(cc.NUM_SLOTS):
        info = scan["slots"].get(i)
        if info:
            text = f"{i}: {info['label']}"
        elif i == scan["orb"]:
            text = f"{i}: orb (skipped)"
        elif i in scan["occupied"]:
            text = f"{i}: occupied, no colour match"
        else:
            text = f"{i}: empty"
        draw.text((i * slot_w + 5, H - 18), text, font=small, fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(30, 30, 40))
    return border(pil)


def stage_masks(scan, scale=0.5):
    cells = []
    for name, mask in scan["masks"].items():
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        w = vis.shape[1]
        for i in range(1, cc.NUM_SLOTS):
            x = i * w // cc.NUM_SLOTS
            cv2.line(vis, (x, 0), (x, vis.shape[0] - 1), (70, 70, 85), 1)
        cells.append(stack([bar(name.upper(), w, 22, size=13), to_pil(vis)], gap=0))

    cols = 2
    cell_w = max(c.width for c in cells)
    cell_h = max(c.height for c in cells)
    rows = (len(cells) + cols - 1) // cols
    out = Image.new("RGB", (cell_w * cols + 8, cell_h * rows + 8), (246, 246, 250))
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        out.paste(cell, (c * (cell_w + 8), r * (cell_h + 8)))
    return out


def heat(pct, ceiling=60.0):
    t = max(0.0, min(1.0, pct / ceiling))
    return (int(240 - 210 * t), int(245 - 185 * t), int(250 - 200 * t))

def stage_occupancy(scan):
    """Table of signal strength per slot: the fastest way to see a threshold problem."""
    cell_w, cell_h, label_w, head_h = 96, 30, 132, 26
    footer_h = 46
    rows = len(scan["masks"]) + 1
    width = label_w + cell_w * cc.NUM_SLOTS
    height = head_h + cell_h * rows + footer_h
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    f, fb = font(13), font(13)

    draw.rectangle([0, 0, width - 1, head_h - 1], fill=(40, 44, 58))
    draw.text((8, 5), "signal % per slot", font=fb, fill=(240, 240, 245))
    for i in range(cc.NUM_SLOTS):
        draw.text((label_w + i * cell_w + cell_w // 2 - 4, 5), str(i), font=fb,
                  fill=(240, 240, 245))

    y = head_h
    for name, mask in scan["masks"].items():
        draw.text((8, y + 7), name, font=fb, fill=(30, 30, 40))
        for i in range(cc.NUM_SLOTS):
            cell = cc._slot_slice(mask, i)
            pct = cv2.countNonZero(cell) / cell.size * 100
            x = label_w + i * cell_w
            draw.rectangle([x, y, x + cell_w - 2, y + cell_h - 2], fill=heat(pct))
            draw.text((x + 8, y + 6), f"{pct:.1f}", font=f, fill=(20, 20, 25))
            if scan["slots"].get(i, {}).get("color") == name:
                draw.rectangle([x, y, x + cell_w - 2, y + cell_h - 2],
                               outline=(20, 120, 60), width=3)
        y += cell_h

    draw.text((8, y + 7), "pill light %", font=fb, fill=(30, 30, 40))
    for i, pct in enumerate(scan["pill_pcts"]):
        x = label_w + i * cell_w
        fill = (205, 235, 210) if pct >= cc.PILL_MIN_FRAC * 100 else (250, 215, 215)
        draw.rectangle([x, y, x + cell_w - 2, y + cell_h - 2], fill=fill)
        draw.text((x + 8, y + 6), f"{pct:.1f}", font=f, fill=(20, 20, 25))
    y += cell_h

    draw.text((8, y + 4),
              f"MIN_SLOT_PCT {cc.MIN_SLOT_PCT}%   PILL_MIN_FRAC {cc.PILL_MIN_FRAC:.0%}   "
              "green box = colour chosen",
              font=font(12), fill=(70, 70, 82))
    draw.text((8, y + 22),
              f"SIZE_SOLIDITY_THRESHOLD {cc.SIZE_SOLIDITY_THRESHOLD}   "
              f"ORB_IS_LAST_ITEM {cc.ORB_IS_LAST_ITEM}",
              font=font(12), fill=(70, 70, 82))
    return border(img)


def stage_tiles(scan, scale=2):
    strip, masks, slots = scan["strip"], scan["masks"], scan["slots"]
    h, w = strip.shape[:2]
    slot_w = w // cc.NUM_SLOTS
    gem_h = int(h * cc.GEM_ZONE_BOTTOM_FRAC)
    if not slots:
        return bar("no gem slots detected (nothing to classify)", 520, height=30)

    panels = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for i, info in sorted(slots.items()):
        tile = strip[:gem_h, i * slot_w:(i + 1) * slot_w]
        vis = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mask = cc._slot_slice(masks[info["color"]], i)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            blob = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(blob)
            blob = (blob.astype(np.float32) * scale).astype(np.int32)
            hull = (hull.astype(np.float32) * scale).astype(np.int32)
            cv2.drawContours(vis, [blob], -1, (70, 210, 90), 2)
            cv2.drawContours(vis, [hull], -1, (60, 120, 255), 2)
        caption = (f"slot {i}  {info['label']}  solidity {info['solidity']} "
                   f"({info['signal_pct']}% colour)")
        panels.append(stack([to_pil(vis), bar(caption, vis.shape[1], 24, size=13)], gap=0))
    return grid(panels, cols=2)


def grid(images, cols=2, gap=8, bg=(246, 246, 250)):
    rows = [images[i:i + cols] for i in range(0, len(images), cols)]
    col_w = [max(img.width for img in row) for row in rows]
    row_h = [max(img.height for img in row) for row in rows]
    out = Image.new("RGB", (sum(col_w) + gap * (cols - 1),
                            sum(row_h) + gap * (len(rows) - 1)), bg)
    y = 0
    for row, height in zip(rows, row_h):
        x = 0
        for col, img in enumerate(row):
            out.paste(img, (x, y))
            x += col_w[col] + gap
        y += height + gap
    return out


def stage_ocr_crops(scan, ocr, scale=0.55):
    """Exactly the pixels EasyOCR is handed, one panel per slot."""
    strip = scan["strip"]
    h, w = strip.shape[:2]
    panels = []
    for i in range(cc.NUM_SLOTS):
        if i in ocr:
            count, raw, processed = ocr[i]
            vis = cv2.resize(processed, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
            rgb = Image.fromarray(vis)          # preprocess already returns RGB
            caption = f"slot {i}: {raw!r} -> {count:,}"
        else:
            rgb = Image.new("RGB", (int(w / cc.NUM_SLOTS * scale), int(h * 0.28 * scale)),
                            (235, 235, 242))
            ImageDraw.Draw(rgb).text((8, 8), "not read", font=font(12), fill=(150, 150, 162))
            caption = f"slot {i}: skipped"
        panels.append(stack([bar(caption, max(rgb.width, 200), 22, size=12), rgb], gap=0))

    width = sum(p.width for p in panels) + 6 * (len(panels) - 1)
    height = max(p.height for p in panels)
    out = Image.new("RGB", (width, height), (246, 246, 250))
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width + 6
    return out


def stage_ocr_table(scan, ocr):
    cols = (60, 190, 220, 130)
    rows = [("slot", "gem", "raw OCR text", "quantity")]
    for i in sorted(ocr):
        count, raw, _processed = ocr[i]
        rows.append((str(i), scan["slots"].get(i, {}).get("label", "?"),
                     repr(raw), f"{count:,}"))
    if len(rows) == 1:
        rows.append(("-", "nothing detected", "-", "-"))

    width, row_h = sum(cols), 28
    img = Image.new("RGB", (width, row_h * len(rows) + 2), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    f = font(13)
    for r, row in enumerate(rows):
        y = r * row_h
        if r == 0:
            draw.rectangle([0, 0, width - 1, row_h - 1], fill=(40, 44, 58))
        x = 0
        for c, text in enumerate(row):
            draw.text((x + 8, y + 6), text, font=f,
                      fill=(240, 240, 245) if r == 0 else (25, 25, 32))
            x += cols[c]
        draw.line([(0, y + row_h - 1), (width, y + row_h - 1)], fill=(225, 225, 232))
    return border(img)


def print_table(scan, ocr):
    header = (f"{'slot':>4}  {'gem':<14} {'signal%':>8} {'solidity':>9} "
              f"{'pill%':>6}  {'raw OCR':<10} {'quantity':>9}")
    print(header)
    print("-" * len(header))
    for i in range(cc.NUM_SLOTS):
        info = scan["slots"].get(i)
        pill = scan["pill_pcts"][i]
        if info:
            count, raw, _p = ocr.get(i, (None, "", None))
            qty = f"{count:,}" if count is not None else "-"
            print(f"{i:>4}  {info['label']:<14} {info['signal_pct']:>8} "
                  f"{info['solidity']:>9} {pill:>6}  {raw[:10]:<10} {qty:>9}")
        else:
            if i == scan["orb"]:
                what = "orb"
            elif i in scan["occupied"]:
                what = "occupied, no colour"
            else:
                what = "empty"
            print(f"{i:>4}  {what:<14} {'':>8} {'':>9} {pill:>6}")
    print(f"\noccupied {scan['occupied']}   orb {scan['orb']}   "
          f"gem slots {scan['gem_indices']}")

# =============================================================================
# DRIVER
# =============================================================================

def debug_image(path, out_root="debug_one", crop=True, zoom=1.0, run_ocr=True):
    scan = scan_image(path, crop=crop)

    ocr = {}
    if run_ocr:
        for i in sorted(scan["slots"]):
            ocr[i] = cc.ocr_slot_quantity(scan["strip"], i)

    out = Path(out_root) / Path(path).stem
    stages = [
        ("01_strip.png", "1. crop + slot grid + zone cuts", stage_strip(scan, zoom)),
        ("02_masks.png", "2. colour masks (gem zone only)", stage_masks(scan)),
        ("03_occupancy.png", "3. per-slot signal (threshold check)", stage_occupancy(scan)),
        ("04_tiles.png", "4. tiles + contour used for solidity", stage_tiles(scan)),
        ("05_ocr_crops.png", "5. what EasyOCR is given", stage_ocr_crops(scan, ocr)),
        ("06_ocr.png", "6. raw OCR -> quantity", stage_ocr_table(scan, ocr)),
    ]

    panels = []
    for filename, title, image in stages:
        save(image, out / filename)
        panels.append(stack([bar(title, image.width, 24, size=14), image], gap=0))
    save(stack(panels, gap=10), out / "07_contact_sheet.png")
    return scan, ocr, out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Dump every stage of the counter for one screenshot.")
    ap.add_argument("images", nargs="+", help="screenshot files or folders")
    ap.add_argument("--out", default="debug_one",
                    help="output folder (default: debug_one)")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="scale of 01_strip.png (default 1.0)")
    ap.add_argument("--no-crop", action="store_true",
                    help="inputs are already cropped strips")
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip EasyOCR (fast, detection only)")
    args = ap.parse_args(argv)

    paths = []
    for src in args.images:
        p = Path(src).expanduser()
        if p.is_dir():
            paths.extend(sorted(cc.find_images(str(p))))
        elif p.exists():
            paths.append(str(p))
        else:
            raise SystemExit(f"Not found: {src}")
    if not paths:
        raise SystemExit("No images found.")

    for n, path in enumerate(paths, 1):
        print(f"\n=== [{n}/{len(paths)}] {path} ===")
        scan, ocr, out = debug_image(path, args.out, crop=not args.no_crop,
                                     zoom=args.zoom, run_ocr=not args.no_ocr)
        print_table(scan, ocr)
        if ocr:
            inventory = {}
            for i, info in scan["slots"].items():
                inventory[info["label"]] = inventory.get(info["label"], 0) + ocr[i][0]
            cc.print_totals(inventory, "This image:")
        print(f"wrote {out}/01_strip.png ... 07_contact_sheet.png")


if __name__ == "__main__":
    main()
