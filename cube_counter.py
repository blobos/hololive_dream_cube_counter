#!/usr/bin/env python3
"""
Gem / cube counter
==================
Reads gem inventory screenshots, detects which of the 6 slots hold a gem (and
its colour), OCRs the quantity printed under each gem, and sums everything up.

Pipeline (each stage is its own function so you can edit/replace it freely):

    expand_archives()      zips / 7z (even nested) -> folders of screenshots
    crop_images()          full screenshots -> cropped strips
    count_gems()           one image -> {"red small": 40, ...}
        apply_color_thresholds()   HSV colour masks             <-- colour tuning lives here
        detect_gems_by_grid()      which slot has which colour
        ocr_all_slots()            read the number under each gem
 dict
    cumulative_gem_count() every image in a folder -> grand total
    add_extra_gems()       optionally type in extra gems by hand

Install:
    pip install easyocr opencv-python numpy Pillow
    pip install pyunpack patool        # only needed for .7z input (.zip needs nothing extra)

Usage (any mix of zips, 7z files and folders; zips inside zips are opened too):
    python cube_counter.py Screenshots.zip
    python cube_counter.py a.zip b.zip some_folder/
    python cube_counter.py cropped_output/ --no-crop # already-cropped images
    python cube_counter.py a.zip --debug             # save masks / OCR crops
    python cube_counter.py a.zip --no-extras
"""

import argparse
import os
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# CONFIG  -- the stuff you're most likely to tweak
# =============================================================================

# Crop box applied to each full screenshot: (left, top, right, bottom)
CROP_BOX = (363, 335, 1557, 560)

# Where things go
EXTRACT_DIR = "extracted_images"
CROPPED_DIR = "cropped_output"
DEBUG_DIR = "debug_output"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
ARCHIVE_EXTENSIONS = (".zip", ".7z")

# Layout of the cropped strip
NUM_SLOTS = 6                 # slots are equal-width columns
MIN_SLOT_PCT = 5.0            # % of a slot's gem area that must match a colour to count as a gem

# Colour-mode layout
GEM_ZONE_BOTTOM_FRAC = 0.70   # colour masks only look at the top 70% (the gem tile), so the
                              # blue quantity digits below can't be mistaken for a blue gem
PILL_ZONE_FRAC = (0.73, 0.89) # vertical span of the quantity pill (used to tell if a slot is occupied)
PILL_MIN_FRAC = 0.5           # fraction of light, unsaturated pixels needed to call a slot occupied
ORB_IS_LAST_ITEM = True       # the last occupied slot is always the orb, not a gem: skip it
SIZE_SOLIDITY_THRESHOLD = 0.92  # shape solidity at/above this = small diamond, below = big star

# OCR region: the bottom strip of each slot, with side margins trimmed
TEXT_ROI_TOP_FRAC = 0.72      # quantity pill sits at ~72-90% of the strip height; 0.80 cut the digits in half
TEXT_ROI_TRIM_X_FRAC = 0.15   # trim 15% off each side to drop the capsule border
OCR_UPSCALE = 3
OCR_ALLOWLIST = "0123456789.kK"
USE_GPU = True                # EasyOCR falls back to CPU on its own if no CUDA
# Text detection (reader.readtext) must stay on. Recognition-only
# (reader.recognize) treats the whole crop as one line of text and reads the
# rounded ends of the quantity pill as extra digits: "325" came back as
# "43250", "820" as "08200", "50" as "6500". Detection crops to the text line
# first and fixes all of them, for ~70 ms per slot instead of ~10 ms.

# How gems are identified:
#   "color"    = HSV colour masks (see get_tuned_gem_bounds); size (small/big) is
#                read from the gem's SHAPE (diamond vs 4-point star).
#   "template" = match each slot against saved gem icons (colour AND size in one
#                step). Needs templates, see make_templates().
DETECTOR = "color"
TEMPLATE_DIR = "gem_templates"
TEMPLATE_MIN_SCORE = 0.85     # 0-1 match score needed to accept a slot as that gem
# Patch of the gem tile used as the template: (x0, y0, x1, y1) in cropped-strip
# pixels, relative to the left edge of slot 0. Inner part of the tile (gem + a bit of background).
SLOT_TEMPLATE_BOX = (33, 10, 153, 150)

# Output ordering / emoji
LABEL_ORDER = [
    "red small", "red big",
    "blue small", "blue big",
    "yellow small", "yellow big",
    "green small", "green big",
]
EMOJI = {
    "red small": "🔴",    "red big": "🟥",
    "blue small": "🔵",   "blue big": "🟦",
    "green small": "🟢",  "green big": "🟩",
    "yellow small": "🟡", "yellow big": "🟨",
}


# =============================================================================
# COLOUR DETECTION  -- edit the bounds here
# =============================================================================

def get_tuned_gem_bounds():
    """
    OpenCV HSV bounds (H: 0-179, S: 0-255, V: 0-255).

    Each colour maps to a LIST of (lower, upper) ranges, so a colour can have
    more than one band. That matters for red, whose hue wraps around 0/179:
    a single range can only cover one side of the wrap.

    High minimum saturation isolates vibrant gems from pale UI backgrounds.
    """
    return {
        "blue":   [([95, 120, 80],  [125, 255, 255])],
        "green":  [([60, 120, 80],  [95, 255, 255])],
        "yellow": [([18, 120, 100], [38, 255, 255])],
        "red":    [([160, 60, 80],  [179, 255, 255])],
        # If red is being missed, it probably also needs the low-hue side of
        # the wrap. Uncomment this second band (and tune it):
        #   ([0, 60, 80], [10, 255, 255]),
    }


def apply_color_thresholds(img_bgr):
    """Returns {color_name: binary_mask} for the strip image."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    masks = {}

    for color_name, ranges in get_tuned_gem_bounds().items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        masks[color_name] = mask

    return masks


# =============================================================================
# SLOT DETECTION  (colour mode)
# =============================================================================

def _slot_slice(mask, slot_index, num_slots=NUM_SLOTS):
    """Returns the vertical column of `mask` belonging to one slot."""
    slot_width = mask.shape[1] // num_slots
    return mask[:, slot_index * slot_width:(slot_index + 1) * slot_width]


def find_occupied_slots(img_bgr, num_slots=NUM_SLOTS):
    """
    Returns the slot indices that hold an item, judged by whether the light
    quantity pill is present under it. Includes the orb.
    """
    h, w, _ = img_bgr.shape
    slot_width = w // num_slots
    y0, y1 = int(h * PILL_ZONE_FRAC[0]), int(h * PILL_ZONE_FRAC[1])
    pad_x = int(slot_width * 0.18)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    occupied = []
    for i in range(num_slots):
        zone = hsv[y0:y1, i * slot_width + pad_x:(i + 1) * slot_width - pad_x]
        light = (zone[..., 1] < 50) & (zone[..., 2] > 190)
        if light.mean() >= PILL_MIN_FRAC:
            occupied.append(i)
    return occupied


def classify_size(slot_mask):
    """
    Decides small vs big from the SHAPE of the gem in a colour mask.
    Small gems are diamonds (convex, solidity ~0.98); big gems are 4-point
    stars (concave, solidity ~0.7-0.85). Raw pixel counts don't separate them
    because the colour masks cover different amounts per colour.
    Returns (size, solidity).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(slot_mask, cv2.MORPH_CLOSE, kernel)  # bridge gaps from patchy highlights
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "small", 0.0
    blob = max(contours, key=cv2.contourArea)
    solidity = cv2.contourArea(blob) / max(cv2.contourArea(cv2.convexHull(blob)), 1)
    return ("small" if solidity >= SIZE_SOLIDITY_THRESHOLD else "big"), round(float(solidity), 2)


def detect_gems_by_grid(masks, slot_indices, min_pct_threshold=MIN_SLOT_PCT, num_slots=NUM_SLOTS):
    """
    For each candidate slot, picks the colour whose mask covers the most of the
    slot, then reads small/big from the shape.
    Returns {slot_index: {"color", "size", "label", "slot_index", "signal_pct", "solidity"}}.
    Occupied slots where no colour matches are reported, not silently dropped:
    that is the sign your colour bounds need tuning.
    """
    slots = {}

    for i in slot_indices:
        best_color, best_pct = None, 0.0
        for color_name, mask in masks.items():
            slot = _slot_slice(mask, i, num_slots)
            pct = cv2.countNonZero(slot) / slot.size * 100
            if pct > best_pct:
                best_color, best_pct = color_name, pct

        if best_color is None or best_pct < min_pct_threshold:
            print(f"  WARNING: slot {i} has an item but no colour matched "
                  f"(best {best_pct:.1f}%) - check get_tuned_gem_bounds()")
            continue

        size, solidity = classify_size(_slot_slice(masks[best_color], i, num_slots))
        slots[i] = {
            "color": best_color,
            "size": size,
            "label": f"{best_color} {size}",
            "slot_index": i,
            "signal_pct": round(best_pct, 1),
            "solidity": solidity,
        }

    return slots


def inspect_slot_percentages(masks, num_slots=NUM_SLOTS):
    """Debug helper: prints the % of matching pixels per slot for each colour.
    Handy when tuning the HSV bounds."""
    for color_name, mask in masks.items():
        pcts = [
            round(cv2.countNonZero(_slot_slice(mask, i, num_slots)) / _slot_slice(mask, i, num_slots).size * 100, 1)
            for i in range(num_slots)
        ]
        print(f"[{color_name.upper()}] Slot %: {pcts}")


# =============================================================================
# TEMPLATE DETECTION  (alternative to colour masks)
# =============================================================================

_templates = None


def make_templates(screenshot_path, labels, templates_dir=TEMPLATE_DIR, crop_box=CROP_BOX):
    """
    One-time setup: saves gem icons from a screenshot as templates.

    labels = what is in each slot, left to right, using None for slots to skip
             (empty slots, the orb, anything that isn't a gem), e.g.
             ["blue small", "blue big", None, "yellow small", "yellow big"]
    Pass crop_box=None if the image is already cropped.
    You need every gem type (4 colours x small/big) saved at least once.
    """
    img = cv2.imread(screenshot_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {screenshot_path}")
    if crop_box is not None:
        l, t, r, b = crop_box
        img = img[t:b, l:r]

    os.makedirs(templates_dir, exist_ok=True)
    slot_width = img.shape[1] // NUM_SLOTS
    x0, y0, x1, y1 = SLOT_TEMPLATE_BOX

    for i, label in enumerate(labels):
        if not label:
            continue
        patch = img[y0:y1, i * slot_width + x0:i * slot_width + x1]
        path = os.path.join(templates_dir, label.replace(" ", "_") + ".png")
        cv2.imwrite(path, patch)
        print(f"Saved template '{label}' -> {path}")


def load_templates(templates_dir=TEMPLATE_DIR):
    """Loads {label: image} from the templates folder (cached after the first call)."""
    global _templates
    if _templates is None:
        _templates = {}
        if os.path.isdir(templates_dir):
            for name in sorted(os.listdir(templates_dir)):
                if name.lower().endswith(".png"):
                    label = os.path.splitext(name)[0].replace("_", " ")
                    _templates[label] = cv2.imread(os.path.join(templates_dir, name))
    if not _templates:
        raise RuntimeError(
            f"No gem templates found in '{templates_dir}/'. Create them once with "
            "make_templates(screenshot, labels), or set DETECTOR = 'color'.")
    return _templates


def match_slot(slot_img, templates):
    """Returns (best_label, best_score) for one slot column."""
    best_label, best_score = None, -1.0
    for label, tpl in templates.items():
        score = float(cv2.matchTemplate(slot_img, tpl, cv2.TM_CCOEFF_NORMED).max())
        if score > best_score:
            best_label, best_score = label, score
    return best_label, best_score


def detect_gems_by_template(img_bgr, min_score=TEMPLATE_MIN_SCORE, num_slots=NUM_SLOTS, verbose=False):
    """
    Matches every slot against the saved gem templates. Slots whose best score
    is below min_score (empty slot, orb tile, ...) are ignored.
    Returns the same slot dicts as detect_gems_by_grid, but already labelled.
    """
    templates = load_templates()
    slot_width = img_bgr.shape[1] // num_slots
    slots = {}

    for i in range(num_slots):
        column = img_bgr[:, i * slot_width:(i + 1) * slot_width]
        label, score = match_slot(column, templates)
        if verbose:
            print(f"  [debug] slot {i}: best '{label}' score {score:.2f}"
                  f"{'' if score >= min_score else '  (below threshold, ignored)'}")
        if score >= min_score:
            color, size = label.split(" ")
            slots[i] = {"color": color, "size": size, "label": label,
                        "slot_index": i, "signal_pct": round(score * 100, 1)}
    return slots


# =============================================================================
# OCR  (quantity under each gem)
# =============================================================================

_reader = None


def get_reader():
    """Creates the EasyOCR reader once, on first use."""
    global _reader
    if _reader is None:
        import easyocr  # imported lazily so the rest of the script works without it
        _reader = easyocr.Reader(["en"], gpu=USE_GPU)
    return _reader


def parse_quantity(text):
    """Converts strings like '40', '325', '1K', '4.1K', '2.5M' into ints."""
    if not text:
        return 0

    cleaned = text.upper().strip().replace(" ", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*([KMB])?", cleaned)
    if not match:
        return 0

    val_str, suffix = match.groups()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return int(round(float(val_str) * multipliers.get(suffix, 1)))


def preprocess_slot_text(img_bgr, slot_index, num_slots=NUM_SLOTS):
    """Crops the quantity text under one slot and prepares it for OCR."""
    h, w, _ = img_bgr.shape
    slot_width = w // num_slots

    roi = img_bgr[int(h * TEXT_ROI_TOP_FRAC):h,
                  slot_index * slot_width:(slot_index + 1) * slot_width]

    # Trim side margins to remove dark capsule borders
    pad_x = int(roi.shape[1] * TEXT_ROI_TRIM_X_FRAC)
    roi = roi[:, pad_x:roi.shape[1] - pad_x]

    roi = cv2.resize(roi, (0, 0), fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)

    # Grayscale + contrast stretch, then back to 3-channel as EasyOCR expects
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def ocr_slot_quantity(img_bgr, slot_index, num_slots=NUM_SLOTS):
    """Reads the quantity for one slot. Returns (count, raw_text, processed_img)."""
    processed = preprocess_slot_text(img_bgr, slot_index, num_slots)

    reader = get_reader()
    results = reader.readtext(
        processed,
        allowlist=OCR_ALLOWLIST,
        detail=0,
        text_threshold=0.5,
        low_text=0.3,
    )
    raw_text = "".join(results)

    if not raw_text:
        # Nothing at all was found: recognition-only mode can still answer,
        # because it has no detector that could have missed the line.
        try:
            raw_text = "".join(reader.recognize(processed, allowlist=OCR_ALLOWLIST,
                                                detail=0))
        except Exception as e:
            print(f"  recognize() fallback failed ({e})")

    return parse_quantity(raw_text), raw_text, processed


def ocr_all_slots(img_bgr, slots, debug_dir=None, tag=""):
    """Adds a 'count' to every detected slot dict (in place) and returns it."""
    for slot_index, info in slots.items():
        count, raw_text, processed = ocr_slot_quantity(img_bgr, slot_index)
        info["count"] = count

        if debug_dir:
            print(f"  [debug] {tag} slot {slot_index}: OCR '{raw_text}' -> {count}")
            save_debug_image(debug_dir, f"{tag}_slot{slot_index}_ocr.png", processed)

    return slots


# =============================================================================
# INVENTORY
# =============================================================================

def build_inventory(slots):
    """Turns labelled slots (after OCR) into {"red small": 4100, ...}.
    If two slots end up with the same label, their counts are added."""
    inventory = {}
    for info in slots.values():
        inventory[info["label"]] = inventory.get(info["label"], 0) + info["count"]
    return slots, inventory


# =============================================================================
# SINGLE-IMAGE PIPELINE
# =============================================================================

def save_debug_image(debug_dir, filename, img):
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, filename), img)


def auto_crop_box(image_path, crop_box=CROP_BOX):
    """
    CROP_BOX for a full screenshot, None for an image that is already a strip.

    Lets every caller (CLI, labeller, tests) feed the pipeline the same way::

        crop = cc.auto_crop_box(path)
        slots, inventory = cc.count_gems(path, crop_box=crop)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    left, top, right, bottom = crop_box
    if img.shape[1] == right - left and img.shape[0] == bottom - top:
        return None
    return crop_box


def count_gems(image_path, crop_box=None, min_pct_threshold=MIN_SLOT_PCT, debug=False):
    """
    Full pipeline for one screenshot.

    crop_box : optional (left, top, right, bottom) applied before processing
    debug    : if True, prints OCR results and saves masks + OCR crops to DEBUG_DIR

    Returns (slots, inventory), e.g. inventory = {"blue small": 40, "blue big": 1000, ...}
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if crop_box is not None:
        l, t, r, b = crop_box
        img = img[t:b, l:r]

    debug_dir = DEBUG_DIR if debug else None
    tag = os.path.splitext(os.path.basename(image_path))[0]

    # 1+2. Which slots contain a gem, and which one
    if debug:
        print(f"[debug] {tag}")
    if DETECTOR == "template":
        slots = detect_gems_by_template(img, verbose=debug)
    else:
        occupied = find_occupied_slots(img)
        if ORB_IS_LAST_ITEM and occupied:
            occupied = occupied[:-1]            # last item is the orb, not a gem
        gem_zone = img[:int(img.shape[0] * GEM_ZONE_BOTTOM_FRAC)]
        masks = apply_color_thresholds(gem_zone)
        slots = detect_gems_by_grid(masks, occupied, min_pct_threshold=min_pct_threshold)
        if debug:
            for color_name, mask in masks.items():
                save_debug_image(debug_dir, f"{tag}_mask_{color_name}.png", mask)
            inspect_slot_percentages(masks)
            for i, info in slots.items():
                print(f"  [debug] slot {i}: {info['label']} "
                      f"(colour {info['signal_pct']}%, solidity {info['solidity']})")

    # 3. OCR the quantity under each detected gem
    slots = ocr_all_slots(img, slots, debug_dir=debug_dir, tag=tag)

    # 4. Build the inventory dict
    return build_inventory(slots)


# =============================================================================
# BATCH PIPELINE  (works on lists of image paths)
# =============================================================================

def find_images(folder):
    """Recursively lists every image under a folder."""
    found = []
    for root, _dirs, filenames in os.walk(folder):
        for filename in sorted(filenames):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                found.append(os.path.join(root, filename))
    return found


def count_gems_in_paths(paths, **kwargs):
    """Runs count_gems on each image path. Returns {path: inventory}."""
    per_image = {}
    for path in paths:
        try:
            _, inventory = count_gems(path, **kwargs)
            per_image[path] = inventory
        except Exception as e:
            print(f"Failed on {path}: {e}")
    return per_image


def cumulative_gem_count(paths, **kwargs):
    """Counts every image and sums the inventories. Returns (totals, per_image)."""
    per_image = count_gems_in_paths(paths, **kwargs)
    total = Counter()
    for inventory in per_image.values():
        total.update(inventory)
    return dict(total), per_image


# =============================================================================
# PREP: ARCHIVES + CROP
# =============================================================================

def is_archive(name):
    return name.lower().endswith(ARCHIVE_EXTENSIONS)


def reset_dir(path):
    """Deletes and recreates a working folder so stale files from earlier runs
    never get counted."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def extract_archive(archive_path, dest_dir):
    """Extracts one archive into dest_dir. .zip uses the standard library;
    anything else (.7z) goes through pyunpack/patool."""
    os.makedirs(dest_dir, exist_ok=True)
    if archive_path.lower().endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    else:
        from pyunpack import Archive  # lazy: only needed for 7z etc.
        Archive(archive_path).extractall(dest_dir)


def expand_archives(sources, extract_dir=EXTRACT_DIR):
    """
    Takes any mix of folders and archives and returns a list of folders that
    contain the actual files. Archives are extracted into extract_dir (one
    subfolder each), and archives found inside folders or inside other
    archives are extracted too.
    """
    folders, queue, seen = [], [], set()

    def queue_archives_in(folder):
        for root, _dirs, filenames in os.walk(folder):
            for filename in filenames:
                if is_archive(filename):
                    queue.append(os.path.join(root, filename))

    for src in sources:
        if os.path.isdir(src):
            folders.append(src)
            queue_archives_in(src)
        elif is_archive(src):
            queue.append(src)
        else:
            print(f"Skipping {src} (not a folder or a supported archive)")

    n = 0
    while queue:
        archive = queue.pop(0)
        key = os.path.abspath(archive)
        if key in seen:
            continue
        seen.add(key)

        n += 1
        stem = os.path.splitext(os.path.basename(archive))[0]
        dest = os.path.join(extract_dir, f"{n:03d}_{stem}")
        print(f"Extracting {archive}...")
        try:
            extract_archive(archive, dest)
        except Exception as e:
            print(f"Failed to extract {archive}: {e}")
            continue

        folders.append(dest)
        queue_archives_in(dest)  # nested archives

    return folders


def crop_images(folders, output_dir=CROPPED_DIR, crop_box=CROP_BOX):
    """
    Crops every image found under `folders` into output_dir (flat).
    Duplicate filenames get a _2, _3... suffix instead of overwriting.
    Returns the list of cropped file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    cropped_paths = []

    for folder in folders:
        for img_path in find_images(folder):
            filename = os.path.basename(img_path)
            stem, ext = os.path.splitext(filename)
            out_path, k = os.path.join(output_dir, filename), 1
            while os.path.exists(out_path):
                k += 1
                out_path = os.path.join(output_dir, f"{stem}_{k}{ext}")
            try:
                with Image.open(img_path) as img:
                    img.crop(crop_box).save(out_path)
                cropped_paths.append(out_path)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")

    print(f"Successfully cropped {len(cropped_paths)} images!")
    return cropped_paths


# =============================================================================
# REPORTING + MANUAL EXTRAS
# =============================================================================

def print_totals(totals, title="Cumulative totals:"):
    """Prints totals in red, blue, yellow, green order (small then big)."""
    print(title)
    extras = [label for label in totals if label not in LABEL_ORDER]
    for label in LABEL_ORDER + sorted(extras):
        if label in totals:
            print(f"  {EMOJI.get(label, '⚪')} {label}: {totals[label]:,}")


def apply_extra_gems(totals, extras):
    """
    Adds manually entered quantities to totals. `extras` is
    {label: "1.5k"} (strings or ints). Blank / unreadable entries are skipped.
    No input() here, so a UI can call it directly.
    """
    totals = dict(totals)
    for label, value in extras.items():
        qty = value if isinstance(value, int) else parse_quantity(str(value))
        if qty:
            totals[label] = totals.get(label, 0) + qty
    return totals


def add_extra_gems(totals):
    """Command-line prompt for an extra quantity per gem type. Enter skips one."""
    print("Enter extra gems to add (press Enter to skip)\n")
    extras = {}

    for label in LABEL_ORDER:
        while True:
            text = input(f"{EMOJI[label]} {label}: ").strip()
            if text == "":
                break
            qty = parse_quantity(text)
            if qty == 0:
                print("  ❌ Couldn't read that, try again (e.g. 40, 1K, 4.1K).")
                continue
            extras[label] = qty
            print(f"  ✅ Added {qty:,}")
            break

    return apply_extra_gems(totals, extras)


# =============================================================================
# MAIN
# =============================================================================

def run_pipeline(sources, crop=True, debug=False):
    """Everything except printing/prompting. Returns (totals, per_image).
    A UI can call this directly."""
    if crop:
        reset_dir(EXTRACT_DIR)
        reset_dir(CROPPED_DIR)
        folders = expand_archives(sources, EXTRACT_DIR)
        paths = crop_images(folders, CROPPED_DIR)
    else:
        reset_dir(EXTRACT_DIR)
        folders = expand_archives(sources, EXTRACT_DIR)
        paths = [p for f in folders for p in find_images(f)]

    if not paths:
        print("No images found.")
        return {}, {}
    return cumulative_gem_count(paths, debug=debug)


def main(argv=None):
    """CLI entry point. In a notebook, call main(["/content/Screenshots.zip"])."""
    parser = argparse.ArgumentParser(description="Count gems from inventory screenshots.")
    parser.add_argument("sources", nargs="*",
                        help="One or more .zip/.7z files and/or folders of images")
    parser.add_argument("--no-crop", action="store_true",
                        help="Images are already cropped; skip the crop step")
    parser.add_argument("--debug", action="store_true",
                        help=f"Print OCR details and save masks/OCR crops to {DEBUG_DIR}/")
    parser.add_argument("--no-extras", action="store_true",
                        help="Skip the manual 'add extra gems' prompt")
    in_notebook = "ipykernel" in sys.modules or "google.colab" in sys.modules
    if argv is None and in_notebook:
        # Colab/Jupyter inject their own args (e.g. `-f kernel.json`); ignore them all.
        args = parser.parse_args([])
    else:
        args = parser.parse_args(argv)

    if not args.sources:
        if in_notebook:
            print('In a notebook, pass your files explicitly, e.g.\n'
                  '    main(["/content/Screenshots.zip"])')
            return
        parser.error("give at least one .zip/.7z file or folder")

    totals, _per_image = run_pipeline(args.sources, crop=not args.no_crop, debug=args.debug)
    print()
    print_totals(totals)

    if not args.no_extras:
        print()
        totals = add_extra_gems(totals)
        print()
        print_totals(totals, "Updated cumulative totals:")


if __name__ == "__main__":
    main()
