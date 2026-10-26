"""Archive handling: .zip via the standard library, .7z via py7zr.

The .7z tests build their own tiny archives, so they need no screenshots and no
7-Zip program installed. That is exactly the regression they guard: patool used
to be the only .7z path, and it fails ("patool can not unpack") on machines
without a 7z/7zz/unar binary.
"""

import os
import zipfile

import cv2
import numpy as np
import pytest

import cube_counter as cc


def write_png(path, value=120):
    """Writes a tiny image and returns its path, as a str for the cc helpers."""
    img = np.full((12, 16, 3), value, np.uint8)
    assert cv2.imwrite(str(path), img)
    return str(path)


def test_extract_zip(tmp_path):
    src = write_png(tmp_path / "shot.png")
    archive = tmp_path / "shots.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(src, arcname=os.path.basename(src))

    dest = tmp_path / "out"
    cc.extract_archive(str(archive), str(dest))

    assert (dest / "shot.png").is_file()


def test_extract_7z_needs_no_external_binary(tmp_path):
    py7zr = pytest.importorskip("py7zr")

    src = write_png(tmp_path / "shot.png")
    archive = tmp_path / "shots.7z"
    with py7zr.SevenZipFile(str(archive), "w") as zf:
        zf.write(src, arcname=os.path.basename(src))

    dest = tmp_path / "out"
    cc.extract_archive(str(archive), str(dest))

    assert (dest / "shot.png").is_file()


def test_expand_archives_unpacks_nested_archives(tmp_path):
    py7zr = pytest.importorskip("py7zr")

    inner_image = write_png(tmp_path / "inner.png")
    inner_zip = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner_zip, "w") as zf:
        zf.write(inner_image, arcname=os.path.basename(inner_image))

    outer_7z = tmp_path / "outer.7z"
    with py7zr.SevenZipFile(str(outer_7z), "w") as zf:
        zf.write(str(inner_zip), arcname=inner_zip.name)

    folders = cc.expand_archives([str(outer_7z)], extract_dir=str(tmp_path / "extract"))
    found = [p for folder in folders for p in cc.find_images(folder)]

    assert [os.path.basename(p) for p in found] == ["inner.png"]