"""Regression test against the hand-labelled answer key.

The key is written by `python label_gems.py --export tests/fixtures/holocube_expected.json`
after the screenshots have been labelled by hand. Point HOLOCUBE_DIR at the
folder holding those screenshots (default ~/Downloads/holocube).
"""

import json
import os
from pathlib import Path

import pytest

import cube_counter as cc

FIXTURE = Path(__file__).parent / "fixtures" / "holocube_expected.json"
IMAGES = Path(os.environ.get("HOLOCUBE_DIR", str(Path.home() / "Downloads" / "holocube")))

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="no answer key yet: run label_gems.py --export tests/fixtures/holocube_expected.json",
)


@pytest.fixture(scope="module")

def answer_key():
    return json.loads(FIXTURE.read_text())



def test_key_is_self_consistent(answer_key):
    per_image = answer_key["per_image"]
    assert answer_key["source_images"] == len(per_image)

    summed = {}
    for values in per_image.values():
        for label, qty in values.items():
            summed[label] = summed.get(label, 0) + qty
    assert answer_key["totals"] == summed


def test_pipeline_matches_the_hand_labels(answer_key):
    pytest.importorskip("easyocr")

    missing = [name for name in answer_key["per_image"] if not (IMAGES / name).exists()]
    if missing:
        pytest.skip(f"{len(missing)} labelled screenshots missing from {IMAGES}")

    problems = []
    for name, human in answer_key["per_image"].items():
        path = str(IMAGES / name)
        _slots, inventory = cc.count_gems(path, crop_box=cc.auto_crop_box(path))
        for label in cc.LABEL_ORDER:
            mine, theirs = inventory.get(label, 0), human.get(label, 0)
            if mine != theirs:
                problems.append(f"{name}  {label:<14} pipeline {mine:>9,}   labelled {theirs:>9,}")

    assert not problems, (f"{len(problems)} value(s) differ from the hand labels:\n"
                          + "\n".join(problems))
