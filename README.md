#Hololive Dreams Cube Counter

Counts gems from inventory screenshots. For each screenshot it finds which of the 6 slots hold a gem, works out the colour and size (small/big), reads the quantity printed under it with OCR, and adds everything up across all your images.

```
🔴 red small: 146,429
🟥 red big: 176
🔵 blue small: 1,767
...
```

## Install

```bash
pip install easyocr opencv-python numpy Pillow
pip install pyunpack patool    # only if you have .7z files (zip needs nothing extra)
```

- EasyOCR downloads its models the first time it runs, so you need internet for the first run.
- It uses the GPU if one is available and falls back to CPU otherwise (set `USE_GPU = False` to force CPU). CPU works but is slower.

## Usage

Pass any mix of `.zip` / `.7z` files and folders. Zips inside zips are opened too.

```bash
python cube_counter.py Screenshots.zip
python cube_counter.py a.zip b.zip some_folder/
python cube_counter.py cropped_output/ --no-crop   # images are already cropped
python cube_counter.py Screenshots.zip --debug     # save masks + OCR crops
python cube_counter.py Screenshots.zip --no-extras # skip the manual-extras prompt
```

| Flag | What it does |
|---|---|
| `--no-crop` | Skip the crop step. Use when your images are already just the gem strip. |
| `--debug` | Prints the per-slot colour match % and OCR reads, and saves masks and OCR crops to `debug_output/`. |
| `--no-extras` | Don't ask for extra gems at the end. |

At the end it asks for extra gems to add by hand (one prompt per gem type, press Enter to skip). Quantities like `40`, `1K`, `4.1K` all work.

## How it works

```
zips / folders
   │  expand_archives()          extract zips (including nested ones)
   ▼
screenshots
   │  crop_images()              crop to the gem strip using CROP_BOX
   ▼
cropped strips
   │  count_gems()               one image at a time:
   │     apply_color_thresholds()    HSV colour masks
   │     detect_gems_by_grid()       which slot has which colour
   │     ocr_all_slots()             read the number under each gem
   │     label_gem_sizes()           small / big, build inventory
   ▼
cumulative_gem_count()           sum all images → totals
   ▼
add_extra_gems()                 optional manual additions
```

Each stage is its own function, so you can edit or replace one without touching the others.

### Assumptions the detection makes

- The cropped strip is split into **6 equal-width slots**.
- Each colour occupies a **small + big pair** of slots. If a colour appears in two slots, the left one is small and the right one is big. If it appears in only one, even slot index = small, odd = big.
- The quantity is printed in the **bottom 20%** of each slot.

If your screenshots are laid out differently, change the config values below.

## Configuration

Everything you're likely to tweak is in the `CONFIG` block at the top of the script.

| Setting | Meaning |
|---|---|
| `CROP_BOX` | `(left, top, right, bottom)` crop applied to full screenshots |
| `NUM_SLOTS` | Number of slot columns (6) |
| `MIN_SLOT_PCT` | % of a slot's pixels that must match a colour to count as a gem |
| `TEXT_ROI_TOP_FRAC` | Where the quantity text starts vertically (0.80 = bottom 20%) |
| `TEXT_ROI_TRIM_X_FRAC` | Side margin trimmed off each slot before OCR (drops the capsule border) |
| `OCR_UPSCALE` | How much the text crop is enlarged before OCR |
| `OCR_ALLOWLIST` | Characters OCR is allowed to return |
| `USE_GPU` | Use the GPU for OCR if available |
| `LABEL_ORDER`, `EMOJI` | Output order and emoji |

## Tuning colours

Colour detection lives in `get_tuned_gem_bounds()`. Values are OpenCV HSV (H: 0–179, S: 0–255, V: 0–255).

Each colour is a **list of ranges**, so a colour can have more than one band:

```python
"red": [
    ([160, 60, 80], [179, 255, 255]),
    ([0, 60, 80],   [10, 255, 255]),   # low side of the hue wraparound
],
```

Red is the tricky one. Its hue sits at both ends of the scale (roughly 0–10 and 160–179), so a single range only catches one side. If red gems are being missed, add the second band (a commented-out example is in the script).

**Workflow for tuning:**

1. Run on a problem screenshot with `--debug`.
2. Open `debug_output/<name>_mask_<color>.png`. White = pixels matched. If a gem isn't white, the bounds are too tight. If background is white, they're too loose.
3. Check the printed `Slot %` lines. A slot needs at least `MIN_SLOT_PCT` to count as a gem.
4. Adjust the bounds and repeat.

## Debugging OCR

With `--debug`, each OCR read prints like `slot 2: OCR '1K' -> 1000`, and the exact image the OCR saw is saved as `debug_output/<name>_slot<N>_ocr.png`. If numbers are misread, look at that image first, then adjust `TEXT_ROI_TOP_FRAC`, `TEXT_ROI_TRIM_X_FRAC`, or `OCR_UPSCALE`.

## Use from Python

The pipeline functions can be imported without running the CLI:

```python
import cube_counter as cc

totals, per_image = cc.run_pipeline(["a.zip", "b.zip"])
totals = cc.apply_extra_gems(totals, {"red small": "2.2k", "blue big": 238})
cc.print_totals(totals)
```

`run_pipeline` never calls `input()` (it only prints progress messages), so it's safe to call from a UI or another script.

## Folders it creates

| Folder | Contents |
|---|---|
| `extracted_images/` | Unpacked archives, one subfolder per archive |
| `cropped_output/` | Cropped strips (duplicate filenames get `_2`, `_3`, …) |
| `debug_output/` | Only with `--debug`: masks and OCR crops |

`extracted_images/` and `cropped_output/` are **deleted and recreated at the start of every run** so old files are never counted. Don't keep anything in them. (With `--no-crop`, `cropped_output/` is left alone, so you can point the script at it.)

## Limitations

- The OCR allowlist is digits, `.`, and `k`/`K`, so quantities with an `M` or `B` suffix won't be read from images. `parse_quantity` does understand `K`/`M`/`B` for manually typed extras.
- If OCR can't read a number, that slot is counted as 0. Use `--debug` to catch these.
- If two colours both match one slot, the stronger match wins.
- Duplicate screenshots are counted twice, so make sure your zips don't overlap.
