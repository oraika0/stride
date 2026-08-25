#!/usr/bin/env python
"""Render `control_delay.pdf` (a 16:9 slide exported from control_delay.pptx)
to a tight-cropped PNG for the paper / HackMD.

Pipeline:
  1. pdftoppm renders the single PDF page to a raster at DPI.
  2. Build a non-white mask (gray < THRESH).
  3. Collapse to a per-row dark-pixel profile, split active rows into segments
     wherever a white gap > GAP_PX appears, and keep ONLY the segment with the
     most ink. This drops stray specks (e.g. a leftover dot near the slide top)
     that are separated from the real figure by a big white band -- a plain
     whitespace trim would instead keep them and leave a tall empty margin.
  4. Within that row band, take the dark columns' bbox, pad, crop, save.

Run (from this dir):
    python crop_control_delay.py
Edit DPI / THRESH / GAP_PX / PAD below to retune.

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/control_delay/crop_control_delay.py

"""
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "control_delay.pdf")
# The caption, like every other generated figure, so it drops into the
# document unrenamed. control_delay.pdf beside it is the slide export this
# reads, not the figure.
OUT = os.path.join(HERE, "Figure 2. Control delay timeline.png")

DPI = 400        # render resolution; content ends up ~2950 px wide
THRESH = 248     # gray < THRESH counts as ink (anti-alias tolerant)
MIN_RUN = 8      # a row/col needs >= this many ink px to count (ignores 1-2px specks)
GAP_PX = 120     # white gap (in px) that separates the figure from stray marks
PAD = 28         # uniform white margin kept around the crop


def render(pdf, dpi):
    with tempfile.TemporaryDirectory() as td:
        stem = os.path.join(td, "page")
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), pdf, stem],
                       check=True)
        png = stem + "-1.png"
        return Image.open(png).convert("RGB").copy()


def main():
    im = render(PDF, DPI)
    dark = np.asarray(im.convert("L")) < THRESH
    rows = dark.sum(axis=1)

    # split active rows into segments separated by white gaps, keep the inkiest
    active = np.where(rows >= MIN_RUN)[0]
    if active.size == 0:
        raise SystemExit("no content found -- check THRESH / the PDF")
    splits = np.where(np.diff(active) > GAP_PX)[0]
    segments = np.split(active, splits + 1)
    seg = max(segments, key=lambda s: rows[s].sum())
    y0, y1 = seg.min(), seg.max() + 1

    cols = dark[y0:y1].sum(axis=0)
    xa = np.where(cols >= MIN_RUN)[0]
    x0, x1 = xa.min(), xa.max() + 1

    W, H = im.size
    box = (max(0, x0 - PAD), max(0, y0 - PAD),
           min(W, x1 + PAD), min(H, y1 + PAD))
    crop = im.crop(box)
    crop.save(OUT, optimize=True)
    dropped = [s for s in segments if s is not seg]
    print(f"render {W}x{H} @ {DPI}dpi -> crop {crop.size}  box={box}")
    if dropped:
        bands = ", ".join(f"y={s.min()}-{s.max()}" for s in dropped)
        print(f"dropped {len(dropped)} stray ink band(s) outside the figure: {bands}")
    print(f"saved {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
