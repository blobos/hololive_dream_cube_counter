# Hololive Dreams Cube Counter

Counts gems from inventory screenshots. For each screenshot it finds which of the 6 slots hold a gem, works out the colour and size (small/big), reads the quantity printed under it with OCR, and adds everything up across all your images.

> **Don't want to set anything up?** Open the notebook in [Google Colab](https://colab.research.google.com/) and run the first 2 cells. You only need your screenshots (or a zip of them).
<img width="920" height="1123" alt="image" src="https://github.com/user-attachments/assets/d44456d6-b894-4d7e-a239-06f8c7772285" />

```
🔴 red small: 146,429
🟥 red big: 176
🔵 blue small: 1,767
...
```

## Install

```bash
pip install easyocr opencv-python numpy Pillow
pip install py7zr              # only if you have .7z files (zip needs nothing extra)
```

`.7z` files are read with [py7zr](https://pypi.org/project/py7zr/), which is pure Python, so no 7-Zip program has to be installed. If py7zr is not available the script falls back to `pyunpack` / `patool`, and those do need an external unpacker (`7z`, `7zz` or `unar`) on your `PATH` — without one they fail with `patool can not unpack`.

- EasyOCR downloads its models the first time it runs, so you need internet for the first run.
- It uses the GPU if one is available and falls back to CPU otherwise (set `USE_GPU = False` to force CPU). CPU works but is slower.

## Capturing screenshots (Windows only, optional)

`1080_macro_scrnsht.ahk` is an [AutoHotkey v2](https://www.autohotkey.com/) script that clicks through the game's pages and takes the screenshots for you. It is **Windows only**: it needs AutoHotkey and Windows' built-in **Win + PrtScn** screenshot shortcut, which saves each screenshot as a PNG in `C:\Users\<you>\Pictures\Screenshots` (named `Screenshot (180).png`, `Screenshot (181).png`, ...).

### Before you run it
- Install AutoHotkey **v2** (the script starts with `#Requires AutoHotkey v2.0`).
- Run the game in **1080p (1920×1080)** and fill the screen. All click positions are absolute screen coordinates recorded at that size, and the counter's `CROP_BOX` and templates assume 1080p screenshots too. A different resolution means clicks land in the wrong place and the crop misses the gems.
- The game window must be titled `hololive-Dreams` (Unity window). If it isn't found, the script silently does nothing.
- Start from the game screen the first click expects, then watch one full cycle to confirm every click lands correctly.

### Controls
| Key | Action |
|---|---|
| `F1` | Start / stop the macro. Stopping takes effect within about 50 ms, between steps. |
| `F6` | Quit the script completely. |

While it runs, a "Macro Started" tooltip stays on screen. It shows up in the screenshots, but in the top-left of the dialog, above the area `CROP_BOX` keeps.

### What one cycle does
Each cycle clicks, in order (names match the comments in the script):

1. `#next` at (1858, 554)
2. `#holomemboard` at (414, 564), as a double-click
3. `#reset` at (534, 988), as a burst of **10–21 quick clicks** (randomised, mostly 10–17)
4. Presses **Win + PrtScn** to save a screenshot of the "Reset the Holomem Board?" dialog
5. `#exit reset` at (1555, 161) to close the dialog, then presses `Esc`

It never clicks the dialog's own Reset button. It only opens the dialog to capture the materials list, then closes it.

Delays between steps are randomised (about 0.8–2 s each), so a cycle takes roughly 6–12 seconds. The mouse glides to each target with a little random jitter.

### Settings at the top of the script
| Setting | Meaning |
|---|---|
| `MouseSpeedMultiplier` | How fast the cursor glides (higher = faster). Default `8`. |
| Click coordinates | The `SmoothClick(x, y)` lines in `PerformMacroTasks()`. Change them if your layout differs. |
| Burst sizes | The `Random(10, 13)` / `Random(14, 17)` / `Random(18, 21)` lines for the `#reset` click. |

### Loop counter
Every cycle adds 1 to a number stored in the Windows registry at `HKEY_CURRENT_USER\SOFTWARE\1080_macro_scrnsht.ahk` (value `i`). It is never reset automatically, so it counts cycles across all runs, not just the current one. Delete that registry value if you want to start from 0.

### Then count the gems
When you've finished, zip (or just point at) the contents of `Pictures\Screenshots` and feed them to the counter:

```bash
python cube_counter.py Screenshots.zip
```

Remove any unrelated screenshots from that folder first, since every image in it is counted.

## Usage

Pass any mix of `.zip` / `.7z` files and folders. Arbitrary nesting is handled too: zips inside zips and archives inside folders are opened as well.

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
   │  expand_archives()          extract archives (including nested ones)
   ▼
screenshots
   │  crop_images()              crop to the gem strip using CROP_BOX
   ▼
cropped strips
   │  count_gems()               one image at a time:
   │     find_occupied_slots()       which slots hold an item (last one = orb, skipped)
   │     apply_color_thresholds()    HSV colour masks (default)
   │     detect_gems_by_grid()       colour per slot + small/big from the gem's shape
   │       or detect_gems_by_template()   match slots to saved gem icons (alternative)
   │     ocr_all_slots()             read the number under each gem
   │     build_inventory()           {"red small": 4100, ...}
   ▼
cumulative_gem_count()           sum all images → totals
   ▼
add_extra_gems()                 optional manual additions
```

Each stage is its own function, so you can edit or replace one without touching the others.

### Assumptions the detection makes

- The cropped strip is split into **6 equal-width slots**.
- The quantity pill sits under each gem, at roughly **72–90%** of the strip height.
- Gems fill slots from the left with no gaps, and **the last item is always the orb**, which is skipped (`ORB_IS_LAST_ITEM`). Slots can shift when a gem type is missing, e.g. blue small, blue big, green small, yellow small, yellow big, orb.
- A slot counts as occupied if its light quantity pill is present.
- **Small vs big comes from shape, not position or pixel count.** Small gems are diamonds (solidity ~0.98), big gems are 4-point stars (~0.7–0.85). Raw pixel counts overlap too much between sizes to be reliable.

If your screenshots are laid out differently, change the config values below.

## Configuration

Everything you're likely to tweak is in the `CONFIG` block at the top of the script.

| Setting | Meaning |
|---|---|
| `CROP_BOX` | `(left, top, right, bottom)` crop applied to full screenshots (set for 1920×1080) |
| `NUM_SLOTS` | Number of slot columns (6) |
| `DETECTOR` | `"color"` (default) or `"template"` |
| `GEM_ZONE_BOTTOM_FRAC` | Colour masks only look at the top part of the strip (the gem tile), so the blue digits can't read as a blue gem |
| `PILL_ZONE_FRAC`, `PILL_MIN_FRAC` | Where to look for the quantity pill, and how much of it must be present, to call a slot occupied |
| `ORB_IS_LAST_ITEM` | Skip the last occupied slot (the orb) |
| `SIZE_SOLIDITY_THRESHOLD` | Shape cutoff: at or above = small diamond, below = big star |
| `TEMPLATE_DIR` | Folder of saved gem icons |
| `TEMPLATE_MIN_SCORE` | Match score (0–1) a slot needs to count as a gem |
| `SLOT_TEMPLATE_BOX` | Which patch of the gem tile is saved as a template |
| `MIN_SLOT_PCT` | `"color"` mode: % of a slot's pixels that must match a colour |
| `TEXT_ROI_TOP_FRAC` | Where the quantity text starts vertically (0.72 = 72% down the strip) |
| `TEXT_ROI_TRIM_X_FRAC` | Side margin trimmed off each slot before OCR (drops the capsule border) |
| `OCR_UPSCALE` | How much the text crop is enlarged before OCR |
| `OCR_ALLOWLIST` | Characters OCR is allowed to return |
| `USE_GPU` | Use the GPU for OCR if available |
| `LABEL_ORDER`, `EMOJI` | Output order and emoji |

## Template setup (optional alternative)

This is only needed if you set `DETECTOR = "template"`. That detector compares each slot with saved icons of the 8 gems (4 colours × small/big). Anything that doesn't match, like an empty slot or the orb tile, is ignored. Create the templates once from screenshots that between them show every gem:

```python
make_templates("Screenshot (180).png", ["red small", "red big", "green small", "green big"])
make_templates("Screenshot (182).png", ["blue small", "blue big", None, "yellow small", "yellow big"])
```

Give the labels left to right, one per slot, and use `None` for slots to skip (empty slots, the orb, a gem you already saved). The templates are saved to `gem_templates/`. In Colab that folder disappears when the runtime resets, so re-run this each session or copy the folder to Google Drive.

Templates assume the same screen resolution and `CROP_BOX` as the screenshots you made them from. If you change resolution, make new templates.

## Tuning colours (`DETECTOR = "color"`)

If an occupied slot matches no colour, the script prints `WARNING: slot N has an item but no colour matched` instead of silently dropping it. That warning is your cue to adjust the bounds below.

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

## Debugging detection

With `--debug` in colour mode, each detected slot prints its colour match % and shape solidity, e.g. `slot 1: red big (colour 15.8%, solidity 0.81)`. If a size is wrong, compare its solidity to `SIZE_SOLIDITY_THRESHOLD`.

In template mode, each slot prints its best match and score, e.g. `slot 4: best 'blue small' score 0.23 (below threshold, ignored)`. Real gems should score close to 1.00 and non-gems well under `TEMPLATE_MIN_SCORE`. If a real gem scores low, make a new template from it.

## Debugging OCR

With `--debug`, each OCR read prints like `slot 2: OCR '1K' -> 1000`, and the exact image the OCR saw is saved as `debug_output/<name>_slot<N>_ocr.png`. If numbers are misread, look at that image first, then adjust `TEXT_ROI_TOP_FRAC`, `TEXT_ROI_TRIM_X_FRAC`, or `OCR_UPSCALE`.

`python debug_one.py <screenshot>` puts every stage of one image side by side (crop, colour masks, per-slot percentages, size contours, the OCR crop, and the raw text), both as separate PNGs and as one contact sheet.

**Don't turn text detection off.** EasyOCR's `recognize()` treats the whole crop as a single line, and the rounded ends of the quantity pill get read as digits: `325` comes back as `43250`, `820` as `08200`, `50` as `6500`. `readtext()` finds the text line first and reads all of them correctly.

## Checking the numbers by hand

`label_gems.py` is a small tkinter window for typing in what the slots actually say, so the pipeline can be checked against something known.

```bash
python label_gems.py                                  # opens on the first unlabelled screenshot
python label_gems.py --build-prefill ~/Downloads/holocube   # cache what the pipeline reads, then exit
python label_gems.py --prefill ocr                    # start with the pipeline's values filled in
python label_gems.py --export tests/fixtures/holocube_expected.json
```

The image is shown at the top with the six slots numbered underneath; below it are the eight gem fields (red/blue/yellow/green × small/big), each with a thumbnail of the tile the pipeline assigned to that label. Type the number you see — `4100`, `4.1K` and `1,800` all work — and leave a field blank when that gem isn't there.

| Key | Action |
|---|---|
| `Tab` / `Shift+Tab` | next / previous field |
| `Enter` | save and go to the next image |
| `Shift+Enter` | save and go to the previous image |
| `Ctrl+E` | this screenshot has no gems: save and move on |
| `Ctrl+K` | skip this screenshot (not counted either way) |
| `F2` | flag it: the detection looks wrong |
| `F3` | show/hide what the pipeline read, next to each field |
| `Ctrl+S` | save now |
| `Ctrl+G` | jump to an image number |
| `Ctrl+=` / `Ctrl+-` | zoom the image |
| `Esc` | save and quit |

Progress goes to `labels.json` on every keystroke, so quitting loses nothing, and `Enter` refuses to advance while a field is unreadable or the image is empty (that's what `Ctrl+E` is for). `--export` writes the answer key and prints where the pipeline and you disagree, value by value.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

The fast tests cover quantity parsing, the colour/size decision (on strips drawn in code, so they need no screenshots), archive extraction (`tests/test_archive.py` builds its own tiny `.zip` / `.7z`, so it needs no 7-Zip program) and the answer-key export. Once `tests/fixtures/holocube_expected.json` exists — that is what `label_gems.py --export` writes — `tests/test_pipeline_fixture.py` runs the real pipeline over the hand-labelled screenshots and fails on any value that disagrees. Set `HOLOCUBE_DIR` if those screenshots aren't in `~/Downloads/holocube`.

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
| `gem_templates/` | Gem icons for template detection (created by `make_templates`, never deleted automatically) |
| `debug_output/` | Only with `--debug`: OCR crops, plus colour masks in colour mode |

`extracted_images/` and `cropped_output/` are **deleted and recreated at the start of every run** so old files are never counted. Don't keep anything in them. (With `--no-crop`, `cropped_output/` is left alone, so you can point the script at it.)

## Speed

OCR is the slow part. On a CPU-only runtime EasyOCR can take seconds per number, and low RAM use is normal because it is compute-bound, not memory-bound. Use a GPU runtime if you can. Measured here on CPU: ~70 ms per number with text detection on (the correct setting) versus ~10 ms with it off (which misreads the pill edges). The first run is also slower because EasyOCR downloads and loads its models once per session.

## Limitations

- The OCR allowlist is digits, `.`, and `k`/`K`, so quantities with an `M` or `B` suffix won't be read from images. `parse_quantity` does understand `K`/`M`/`B` for manually typed extras.
- If OCR can't read a number, that slot is counted as 0. Use `--debug` to catch these.
- If two colours both match one slot, the stronger match wins.
- Duplicate screenshots are counted twice, so make sure your zips don't overlap.
