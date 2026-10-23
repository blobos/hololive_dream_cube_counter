#!/usr/bin/env python3
"""
Ground-truth labeler for gem inventory screenshots
==================================================
Shows one cropped inventory strip at a time and lets you type the quantity
printed under each gem. The result is the hand-verified answer key the tests
compare the pipeline against, so a wrong count can be traced to colour
detection, cropping, or OCR instead of guesswork.

This is a dev tool only: cube_counter.py never imports it. It needs just the
detection stage (opencv + numpy + Pillow). EasyOCR is used only by
--build-prefill, which records what the pipeline reads so you can diff it.

Usage:
    python label_gems.py ~/Downloads/holocube
    python label_gems.py ~/Downloads/holocube --prefill ocr
    python label_gems.py --build-prefill ~/Downloads/holocube
    python label_gems.py --export tests/fixtures/holocube_expected.json

Keys:
    Tab / Shift+Tab   next / previous field
    Enter             save + next image
    Shift+Enter       save + previous image (values reload; edits overwrite)
    F2                flag this image ("detection looks wrong")
    F3                show/hide what the pipeline read (hidden by default)
    Ctrl+E            mark this image as empty, then next
    Ctrl+K            skip this image
    Ctrl+G            go to image number
    Ctrl+S            save now
    Ctrl+Backspace    clear the focused field
    Ctrl+= / Ctrl+-   zoom the strip
    Esc               save + quit
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import simpledialog, ttk

import cube_counter as cc


HERE = Path(__file__).resolve().parent
DEFAULT_STATE = str(HERE / "labels.json")
PREFILL_FILE = HERE / "labels.ocr_prefill.json"
DEFAULT_SOURCE = Path.home() / "Downloads" / "holocube"

LABELS = list(cc.LABEL_ORDER)
DEFAULT_ZOOM = 1.6
THUMB_SIZE = (76, 60)

COLORS = {"red": "#d93a4f", "blue": "#2f6fd0", "yellow": "#a87900", "green": "#1f8a4c"}
BG_NORMAL = "white"
BG_REVIEWED = "#e9f8ec"
BG_BAD = "#ffe9e9"
BG_THUMB = "#f2f2f6"

VALUE_RE = re.compile(r"^\d[\d,]*(?:\.\d+)?\s*[kKmMbB]?$")

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


# =============================================================================
# SMALL HELPERS
# =============================================================================

def natural_key(text):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def parse_value(text):
    """'4100' / '4.1K' / '1,800' -> int; '' -> 0; anything else -> None."""
    raw = str(text).strip()
    if not raw:
        return 0
    if not VALUE_RE.match(raw):
        return None
    return cc.parse_quantity(raw.replace(",", ""))


def pil_font(size):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_state(path):
    state = {"version": 1, "images": {}}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            state.update(json.load(fh))
    state.setdefault("images", {})
    return state


def save_state(state, path):
    atomic_write_json(path, state)


def collect_images(sources):
    """[(display_name, path)] for every image under the given files/folders."""
    paths = []
    for src in sources:
        p = Path(src).expanduser()
        if p.is_dir():
            paths.extend(cc.find_images(str(p)))
        elif p.is_file():
            paths.append(str(p))
        else:
            raise SystemExit(f"Not found: {src}")

    paths = sorted(dict.fromkeys(paths), key=lambda q: natural_key(os.path.basename(q)))
    names = [os.path.basename(q) for q in paths]
    counts = Counter(names)
    out = []
    for path, name in zip(paths, names):
        display = name if counts[name] == 1 else str(Path(path).relative_to(Path(path).anchor))
        out.append((display, path))
    return out


def load_strip(path, crop=True):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Could not read {path}")
    l, t, r, b = cc.CROP_BOX
    if not crop or (img.shape[1] == r - l and img.shape[0] == b - t):
        return img                      # already a cropped strip
    if img.shape[0] < b or img.shape[1] < r:
        raise SystemExit(f"{path} is {img.shape[1]}x{img.shape[0]}, "
                         f"smaller than CROP_BOX {r}x{b}")
    return img[t:b, l:r]


def analyse(strip):
    """Detection only (no OCR): which slots are occupied and what they look like."""
    with contextlib.redirect_stdout(io.StringIO()):
        occupied = cc.find_occupied_slots(strip)
        orb = occupied[-1] if (cc.ORB_IS_LAST_ITEM and occupied) else None
        gems = occupied[:-1] if orb is not None else occupied
        zone = strip[:int(strip.shape[0] * cc.GEM_ZONE_BOTTOM_FRAC)]
        masks = cc.apply_color_thresholds(zone)
        slots = cc.detect_gems_by_grid(masks, gems)

    by_label = {}
    for i, info in sorted(slots.items()):
        by_label.setdefault(info["label"], []).append(i)

    return {"strip": strip, "occupied": occupied, "orb": orb, "slots": slots,
            "by_label": by_label, "masks": masks, "gems": gems}


# =============================================================================
# DRAWING
# =============================================================================

def render_strip(strip, detection, zoom=DEFAULT_ZOOM, show_pipeline=False,
                 thumb_size=(84, 66)):
    """The strip, zoomed, with slot numbers drawn underneath the tile row."""
    h, w = strip.shape[:2]
    slot_w = w // cc.NUM_SLOTS

    big = cv2.resize(strip, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    W, H = pil.size

    orb = detection["orb"]
    if orb is not None:
        x0 = int(orb * slot_w * zoom)
        draw.rectangle([x0 + 2, 2, x0 + int(slot_w * zoom) - 3, H - 3],
                       outline=(130, 130, 145), width=2)

    for i in range(1, cc.NUM_SLOTS):
        x = int(i * slot_w * zoom)
        draw.line([(x, 0), (x, H)], fill=(90, 90, 105), width=1)

    font = pil_font(max(11, int(12 * zoom)))
    for i in range(cc.NUM_SLOTS):
        info = detection["slots"].get(i)
        if info is not None and show_pipeline:
            text = f"{i} {info['label']}"
        elif i == orb and show_pipeline:
            text = f"{i} orb"
        else:
            text = str(i)
        draw.text((int(i * slot_w * zoom) + int(4 * zoom), H - int(20 * zoom)),
                  text, font=font, fill=(245, 245, 250),
                  stroke_width=2, stroke_fill=(35, 35, 50))
    return pil


def slot_thumbnail(strip, slot, size=THUMB_SIZE):
    """The gem tile of one slot, for the field next to it."""
    h, w = strip.shape[:2]
    slot_w = w // cc.NUM_SLOTS
    tile = strip[:int(h * cc.GEM_ZONE_BOTTOM_FRAC), slot * slot_w:(slot + 1) * slot_w]
    pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
    return pil.resize(size, Image.LANCZOS)


def placeholder_thumbnail(size=THUMB_SIZE, text="-"):
    pil = Image.new("RGB", size, (238, 238, 244))
    draw = ImageDraw.Draw(pil)
    font = pil_font(20)
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((size[0] - box[2]) / 2, (size[1] - box[3]) / 2 - 5),
              text, font=font, fill=(165, 165, 180))
    return pil


# =============================================================================
# THE WINDOW
# =============================================================================

class LabelApp(tk.Tk):
    """One strip at a time, eight quantity fields, keyboard driven."""

    def __init__(self, images, state_path=DEFAULT_STATE, prefill_mode="blank",
                 prefill=None, zoom=DEFAULT_ZOOM, start=0, crop=True):
        super().__init__()
        self.title("Gem ground truth")
        self.geometry("+40+40")

        self.images = list(images)
        self.state_path = state_path
        self.state = load_state(state_path)
        self.prefill = prefill or {}
        self.prefill_mode = prefill_mode
        self.crop = crop
        self.zoom = float(zoom)
        self.show_pipeline = False
        self.index = max(0, min(int(start), len(self.images) - 1))
        self._det_cache = {}
        self._save_job = None
        self._flash_job = None
        self._photo = None
        self._thumbs = []
        self.detection = None

        self._build_ui()
        self._bind_keys()
        self._show()
        self._fit_window()

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        self.configure(padx=10, pady=8)

        head = tk.Frame(self)
        head.pack(fill="x")
        self.name_lbl = tk.Label(head, text="", font=("TkDefaultFont", 13, "bold"),
                                 anchor="w")
        self.name_lbl.pack(side="left")
        self.pos_lbl = tk.Label(head, text="", fg="#666")
        self.pos_lbl.pack(side="left", padx=12)
        self.flag_lbl = tk.Label(head, text="", fg="#b36b00",
                                 font=("TkDefaultFont", 11, "bold"))
        self.flag_lbl.pack(side="right")
        self.reviewed_lbl = tk.Label(head, text="", fg="#1f8a4c")
        self.reviewed_lbl.pack(side="right", padx=12)

        self.image_lbl = tk.Label(self, bd=1, relief="solid", bg="#222")
        self.image_lbl.pack(pady=(6, 8))

        body = tk.Frame(self)
        body.pack()

        self.fields = {}
        for idx, label in enumerate(LABELS):
            row, column = divmod(idx, 4)
            cell = tk.Frame(body, padx=6, pady=4)
            cell.grid(row=row, column=column, sticky="w")

            thumb = tk.Label(cell, bg=BG_THUMB, bd=1, relief="solid")
            thumb.grid(row=0, column=0, rowspan=2, padx=(0, 8))

            name = tk.Label(cell, text=label, fg=COLORS[label.split()[0]],
                            font=("TkDefaultFont", 11, "bold"))
            name.grid(row=0, column=1, sticky="w")

            hint = tk.Label(cell, text="", fg="#999", font=("TkDefaultFont", 9))
            hint.grid(row=0, column=2, sticky="w", padx=(6, 0))

            var = tk.StringVar()
            entry = tk.Entry(cell, textvariable=var, width=9, justify="right",
                             font=("TkDefaultFont", 15), bd=1, relief="solid")
            entry.grid(row=1, column=1, sticky="w")

            parsed = tk.Label(cell, text="", fg="#888", width=10, anchor="w",
                              font=("TkDefaultFont", 10))
            parsed.grid(row=1, column=2, sticky="w", padx=(6, 0))

            var.trace_add("write", lambda *_a, l=label: self._on_edit(l))
            entry.bind("<KeyRelease>", lambda _e: self._schedule_save())
            self.fields[label] = {"thumb": thumb, "entry": entry, "var": var,
                                  "parsed": parsed, "hint": hint, "name": name}

        foot = tk.Frame(self)
        foot.pack(fill="x", pady=(10, 0))
        ttk.Button(foot, text="< Back  (Shift+Enter)", command=self.prev_image,
                   takefocus=0).pack(side="left")
        ttk.Button(foot, text="Next >  (Enter)", command=self.next_image,
                   takefocus=0).pack(side="left", padx=6)
        ttk.Button(foot, text="Flag (F2)", command=self.toggle_flag,
                   takefocus=0).pack(side="left")
        self.status_lbl = tk.Label(foot, text="", anchor="e", fg="#444")
        self.status_lbl.pack(side="right")

        tk.Label(self, fg="#888", font=("TkDefaultFont", 9), justify="left",
                 text="Tab next field - Enter next image - Shift+Enter back - "
                      "F3 pipeline values - Ctrl+E empty - Ctrl+K skip - Esc quit"
                 ).pack(anchor="w", pady=(6, 0))

    def _bind_keys(self):
        self.bind("<Return>", self._on_enter)
        self.bind("<KP_Enter>", self._on_enter)
        self.bind("<Shift-Return>", self._on_shift_enter)
        self.bind("<F2>", self._on_flag_key)
        self.bind("<F3>", self._on_pipeline_key)
        for seq in ("<Control-s>", "<Control-S>"):
            self.bind(seq, self._on_save_key)
        for seq in ("<Control-e>", "<Control-E>"):
            self.bind(seq, self._on_empty_key)
        for seq in ("<Control-k>", "<Control-K>"):
            self.bind(seq, self._on_skip_key)
        for seq in ("<Control-g>", "<Control-G>"):
            self.bind(seq, self._on_goto_key)
        self.bind("<Control-plus>", lambda _e: self._zoom_by(0.2))
        self.bind("<Control-equal>", lambda _e: self._zoom_by(0.2))
        self.bind("<Control-minus>", lambda _e: self._zoom_by(-0.2))
        self.bind("<Control-BackSpace>", self._on_clear_key)
        self.bind("<Escape>", self._on_quit_key)
        self.protocol("WM_DELETE_WINDOW", self.quit_app)

    # ------------------------------------------------------------ key events
    def _on_enter(self, _event=None):
        self.next_image()
        return "break"

    def _on_shift_enter(self, _event=None):
        self.prev_image()
        return "break"

    def _on_flag_key(self, _event=None):
        self.toggle_flag()
        return "break"

    def _on_pipeline_key(self, _event=None):
        self.toggle_pipeline_view()
        return "break"

    def _on_save_key(self, _event=None):
        self.save_now()
        return "break"

    def _on_empty_key(self, _event=None):
        self.mark_empty_next()
        return "break"

    def _on_skip_key(self, _event=None):
        self.skip_next()
        return "break"

    def _on_goto_key(self, _event=None):
        self.goto_image()
        return "break"

    def _on_clear_key(self, _event=None):
        widget = self.focus_get()
        if isinstance(widget, tk.Entry):
            widget.delete(0, "end")
        return "break"

    def _on_quit_key(self, _event=None):
        self.quit_app()
        return "break"

    # ------------------------------------------------------------- bookkeeping
    @property
    def current_name(self):
        return self.images[self.index][0]

    def record(self, name=None):
        return self.state["images"].setdefault(name or self.current_name, {})

    def detection_for(self, name, path):
        if name not in self._det_cache:
            self._det_cache[name] = analyse(load_strip(path, crop=self.crop))
        return self._det_cache[name]

    def prefill_for(self, name):
        entries = self.prefill.get("images", self.prefill)
        return entries.get(name) or {}

    def collect_values(self):
        """({label: quantity}, [labels that couldn't be parsed])."""
        values, bad = {}, []
        for label in LABELS:
            qty = parse_value(self.fields[label]["var"].get())
            if qty is None:
                bad.append(label)
            elif qty:
                values[label] = qty
        return values, bad

    def save(self):
        self._save_job = None
        save_state(self.state, self.state_path)

    def _schedule_save(self):
        if self._save_job is not None:
            self.after_cancel(self._save_job)
        self._save_job = self.after(500, self._autosave)

    def _autosave(self):
        self._save_job = None
        values, _bad = self.collect_values()
        rec = self.record()
        if not rec.get("reviewed"):
            rec["draft"] = values
        save_state(self.state, self.state_path)

    # -------------------------------------------------------------- showing
    def _show(self):
        name, path = self.images[self.index]
        self.detection = self.detection_for(name, path)
        rec = self.record(name)
        saved = rec.get("values") or {}
        draft = rec.get("draft") or {}
        pipeline = self.prefill_for(name) if self.prefill_mode == "ocr" else {}

        for label in LABELS:
            if label in saved:
                text = str(saved[label])
            elif label in draft:
                text = str(draft[label])
            elif pipeline.get(label):
                text = str(pipeline[label])
            else:
                text = ""
            self.fields[label]["var"].set(text)

        self._render_image()
        self._render_thumbs()
        self._update_header()
        self.status_lbl.configure(text="")

        empty = [l for l in LABELS if not self.fields[l]["var"].get()]
        entry = self.fields[empty[0] if empty else LABELS[0]]["entry"]
        entry.focus_set()
        entry.select_range(0, "end")

    def _render_image(self):
        pil = render_strip(self.detection["strip"], self.detection, self.zoom,
                           self.show_pipeline)
        self._photo = ImageTk.PhotoImage(pil)
        self.image_lbl.configure(image=self._photo)

    def _render_thumbs(self):
        strip = self.detection["strip"]
        self._thumbs = []
        for label in LABELS:
            slots = self.detection["by_label"].get(label) or []
            pil = slot_thumbnail(strip, slots[0]) if slots else placeholder_thumbnail()
            photo = ImageTk.PhotoImage(pil)
            self._thumbs.append(photo)
            self.fields[label]["thumb"].configure(image=photo)

    def _update_header(self):
        rec = self.record()
        self.name_lbl.configure(text=self.current_name)
        self.pos_lbl.configure(text=f"image {self.index + 1} / {len(self.images)}")
        reviewed = sum(1 for r in self.state["images"].values() if r.get("reviewed"))
        flagged = sum(1 for r in self.state["images"].values() if r.get("flag"))
        extra = f"  -  flagged {flagged}" if flagged else ""
        self.reviewed_lbl.configure(
            text=f"reviewed {reviewed}/{len(self.images)}{extra}")
        self.flag_lbl.configure(text="FLAGGED" if rec.get("flag") else "")

    def _fit_window(self):
        """Shrinks the strip if the window would not fit on screen."""
        self.update_idletasks()
        over = self.winfo_reqwidth() - (self.winfo_screenwidth() - 60)
        if over > 0:
            strip_w = max(1, self.detection["strip"].shape[1])
            self.zoom = max(0.8, round(self.zoom - over / strip_w, 2))
            self._render_image()

    # --------------------------------------------------------------- editing
    def _on_edit(self, label):
        field = self.fields[label]
        qty = parse_value(field["var"].get())
        if qty is None:
            field["parsed"].configure(text="bad value", fg="#c00")
            field["entry"].configure(bg=BG_BAD)
        else:
            field["parsed"].configure(text=f"= {qty:,}" if qty else "", fg="#888")
            reviewed = self.record().get("reviewed")
            field["entry"].configure(bg=BG_REVIEWED if reviewed else BG_NORMAL)
        self._update_hint(label)

    def _update_hint(self, label):
        hint = self.fields[label]["hint"]
        if not self.show_pipeline:
            hint.configure(text="")
            return
        pipeline = self.prefill_for(self.current_name).get(label, 0)
        mine = parse_value(self.fields[label]["var"].get()) or 0
        if not pipeline and not mine:
            hint.configure(text="pipeline: -", fg="#aaa")
        elif pipeline == mine:
            hint.configure(text=f"pipeline: {pipeline:,} ok", fg="#1f8a4c")
        else:
            hint.configure(text=f"pipeline: {pipeline:,}", fg="#c2410c")

    # ------------------------------------------------------------ navigation
    def commit(self):
        values, bad = self.collect_values()
        if bad:
            self.flash("Can't read: " + ", ".join(bad), error=True)
            self.fields[bad[0]]["entry"].focus_set()
            return False
        rec = self.record()
        if not values and not rec.get("empty_confirmed"):
            self.flash("All fields empty - Ctrl+E marks the image empty", error=True)
            return False
        rec["values"] = values
        rec["reviewed"] = True
        rec["skipped"] = False
        rec["detected"] = {str(i): info["label"]
                           for i, info in self.detection["slots"].items()}
        rec["orb_slot"] = self.detection["orb"]
        rec.pop("draft", None)
        self.save()
        return True

    def next_image(self):
        if not self.commit():
            return
        if self.index >= len(self.images) - 1:
            self.flash("Last image - Esc saves and quits")
            return
        self.index += 1
        self._show()

    def prev_image(self):
        # Going back never needs a complete answer: keep what is typed as a draft.
        values, bad = self.collect_values()
        if values and not bad:
            self.commit()
        else:
            self.record()["draft"] = values
            self.save()
        if self.index == 0:
            self.flash("First image")
            return
        self.index -= 1
        self._show()

    def toggle_flag(self):
        rec = self.record()
        rec["flag"] = not rec.get("flag", False)
        self.save()
        self._update_header()
        self.flash("Flagged - detection looks wrong" if rec["flag"] else "Flag cleared")

    def toggle_pipeline_view(self):
        self.show_pipeline = not self.show_pipeline
        self._render_image()
        for label in LABELS:
            self._update_hint(label)
        if self.show_pipeline and not self.prefill_for(self.current_name):
            self.flash("No pipeline values cached - run --build-prefill", error=True)
        else:
            self.flash("Pipeline values shown" if self.show_pipeline
                       else "Pipeline values hidden")

    def mark_empty_next(self):
        self.record()["empty_confirmed"] = True
        for label in LABELS:
            self.fields[label]["var"].set("")
        if not self.commit():
            return
        if self.index < len(self.images) - 1:
            self.index += 1
            self._show()
        else:
            self.flash("Last image - Esc saves and quits")

    def skip_next(self):
        rec = self.record()
        rec["skipped"] = True
        rec["reviewed"] = False
        rec.pop("draft", None)
        self.save()
        if self.index < len(self.images) - 1:
            self.index += 1
            self._show()
        else:
            self.flash("Last image - Esc saves and quits")

    def goto_image(self):
        if not self.commit():
            return
        target = simpledialog.askinteger(
            "Go to image", f"Image number (1-{len(self.images)}):", parent=self,
            minvalue=1, maxvalue=len(self.images))
        if target:
            self.index = target - 1
            self._show()

    def save_now(self):
        values, bad = self.collect_values()
        self.record()["draft"] = values
        self.save()
        self.flash("Saved" if not bad else "Saved, but some values are unreadable",
                   error=bool(bad))

    def _zoom_by(self, delta):
        self.zoom = max(0.8, min(3.0, round(self.zoom + delta, 2)))
        self._render_image()
        self.flash(f"zoom {self.zoom:g}x")
        return "break"

    def quit_app(self, _event=None):
        """Esc / window close: keep whatever is typed, then exit."""
        values, bad = self.collect_values()
        if values and not bad:
            self.commit()
        else:
            self.record()["draft"] = values
            self.save()
        self.destroy()

    def flash(self, text, error=False):
        self.status_lbl.configure(text=text, fg="#b91c1c" if error else "#1f8a4c")
        if self._flash_job is not None:
            self.after_cancel(self._flash_job)
        self._flash_job = self.after(3000, lambda: self.status_lbl.configure(text=""))


# =============================================================================
# WHAT THE PIPELINE READS (prefill) AND THE ANSWER KEY (export)
# =============================================================================

def load_prefill(path=PREFILL_FILE):
    if not os.path.exists(path):
        return {"images": {}}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault("images", {})
    return data


def build_prefill(images, out_path=PREFILL_FILE):
    """Runs the real pipeline once per image and caches its answer (needs EasyOCR)."""
    cache = load_prefill(out_path)
    for n, (name, path) in enumerate(images, 1):
        cached = cache["images"].get(name)
        if cached is not None and "error" not in cached:
            continue                                  # retry images that failed
        print(f"[{n}/{len(images)}] {name}", flush=True)
        try:
            _slots, inventory = cc.count_gems(str(path),
                                              crop_box=cc.auto_crop_box(str(path)))
        except Exception as exc:                        # keep going, note the failure
            print(f"  failed: {exc}")
            cache["images"][name] = {"error": str(exc)}
        else:
            cache["images"][name] = dict(inventory)
        atomic_write_json(out_path, cache)
    print(f"Pipeline values cached in {out_path}")
    return cache


def export_labels(state, out_path, prefill=None):
    """Writes the verified answer key and prints the pipeline-vs-human diff."""
    per_image, flagged, skipped, unreviewed = {}, [], [], []
    for name, rec in state["images"].items():
        if rec.get("skipped"):
            skipped.append(name)
            continue
        if not rec.get("reviewed"):
            unreviewed.append(name)
            continue
        per_image[name] = dict(rec.get("values") or {})
        if rec.get("flag"):
            flagged.append(name)

    totals = Counter()
    for values in per_image.values():
        totals.update(values)
    ordered = {label: totals[label] for label in LABELS if label in totals}
    for label in sorted(set(totals) - set(LABELS)):
        ordered[label] = totals[label]

    payload = {"source_images": len(per_image), "per_image": per_image,
               "totals": ordered, "flagged": flagged,
               "unreviewed": unreviewed, "skipped": skipped}
    atomic_write_json(out_path, payload)

    print(f"Wrote {out_path}")
    print(f"  {len(per_image)} reviewed, {len(unreviewed)} not reviewed, "
          f"{len(skipped)} skipped, {len(flagged)} flagged")
    cc.print_totals(ordered, "Verified totals:")

    if flagged:
        print("\nFlagged while reviewing:")
        for name in flagged:
            rec = state["images"][name]
            print(f"  {name}: you={rec.get('values')} pipeline_slots={rec.get('detected')}")

    if prefill:
        entries = prefill.get("images", prefill)
        diffs, compared, no_data = [], 0, 0
        for name, values in per_image.items():
            pipe = entries.get(name) or {}
            if not pipe or "error" in pipe:
                no_data += 1
                continue
            for label in LABELS:
                human, machine = values.get(label, 0), pipe.get(label, 0)
                compared += 1
                if human != machine:
                    diffs.append((name, label, machine, human))
        agree = compared - len(diffs)
        pct = agree / compared * 100 if compared else 0.0
        print(f"\nPipeline vs hand-verified: {agree}/{compared} values agree "
              f"({pct:.1f}%)" + (f", {no_data} images without pipeline data" if no_data else ""))
        for name, label, machine, human in diffs:
            print(f"  {name}  {label:14s} pipeline {machine:>9,}   you {human:>9,}")
    return payload


# =============================================================================
# COMMAND LINE
# =============================================================================

def first_unreviewed(state, images):
    for i, (name, _path) in enumerate(images):
        if not state["images"].get(name, {}).get("reviewed"):
            return i
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hand-verify gem quantities and build the answer key.")
    parser.add_argument("sources", nargs="*",
                        help=f"folders or files of screenshots (default: {DEFAULT_SOURCE})")
    parser.add_argument("--state", default=DEFAULT_STATE,
                        help="progress file, written on every keystroke "
                             "(default: labels.json)")
    parser.add_argument("--prefill", choices=("blank", "ocr"), default="blank",
                        help="start blank (default) or pre-filled from the pipeline")
    parser.add_argument("--build-prefill", action="store_true",
                        help="run the pipeline on every image and cache what it reads")
    parser.add_argument("--export", metavar="PATH",
                        help="write the answer key JSON and exit")
    parser.add_argument("--zoom", type=float, default=DEFAULT_ZOOM)
    parser.add_argument("--start", type=int, default=0,
                        help="1-based image number to open "
                             "(default: first one not reviewed yet)")
    parser.add_argument("--no-crop", action="store_true",
                        help="images are already cropped strips")
    args = parser.parse_args(argv)

    if args.export:
        prefill = load_prefill() if os.path.exists(PREFILL_FILE) else None
        export_labels(load_state(args.state), args.export, prefill=prefill)
        return

    sources = args.sources or ([str(DEFAULT_SOURCE)] if DEFAULT_SOURCE.is_dir() else [])
    if not sources:
        parser.error("give a folder of screenshots, e.g. ~/Downloads/holocube")

    images = collect_images(sources)
    if not images:
        raise SystemExit(f"No images found in {sources}")

    prefill = load_prefill()
    if args.build_prefill:
        build_prefill(images)
        print("Run it again with --prefill ocr to review against those values.")
        return
    if args.prefill == "ocr" and not prefill["images"]:
        raise SystemExit("--prefill ocr needs cached values: run --build-prefill first")

    state = load_state(args.state)
    start = args.start - 1 if args.start else first_unreviewed(state, images)
    app = LabelApp(images, state_path=args.state, prefill_mode=args.prefill,
                   prefill=prefill, zoom=args.zoom, start=start,
                   crop=not args.no_crop)
    app.mainloop()

    reviewed = sum(1 for r in app.state["images"].values() if r.get("reviewed"))
    print(f"\nSaved {args.state}: {reviewed}/{len(images)} reviewed")
    flagged = sorted(n for n, r in app.state["images"].items() if r.get("flag"))
    if flagged:
        print(f"Flagged ({len(flagged)}): " + ", ".join(flagged))
    print("Export: python " + Path(__file__).name
          + " --export tests/fixtures/holocube_expected.json")


if __name__ == "__main__":
    main()
