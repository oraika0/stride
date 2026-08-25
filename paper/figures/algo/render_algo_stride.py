#!/usr/bin/env python
"""Compile `_algo_stride_overleaf.md` (a LaTeX algorithm doc) and crop it to
`algo_stride.png` for the paper / HackMD.

Edit `_algo_stride_overleaf.md`, run this,
re-upload algo_stride.png to HackMD ([algo_stride] in main_zh.md).

Pipeline:
  1. pdflatex compiles the .md (it is really a .tex) in a temp dir.
  2. pdftoppm rasterizes the single page at DPI.
  3. Build a non-white mask (gray < THRESH), take its bbox, pad, crop, save.
     The algorithm is one ruled block so a plain whitespace trim is enough;
     MIN_RUN ignores stray 1-2px specks just in case.

Needs: pdflatex + pdftoppm on PATH, plus numpy/Pillow (conda stride has them).
Tune DPI / THRESH / PAD below.

Run:

    cd ~/stride
    conda activate stride
    python paper/figures/algo/render_algo_stride.py

"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "_algo_stride_overleaf.md")   # LaTeX source (.tex content)
OUT = os.path.join(HERE, "algo_stride.png")

DPI = 600        # render resolution; content ends up ~3400 px wide
THRESH = 248     # gray < THRESH counts as ink (anti-alias tolerant)
MIN_RUN = 3      # a row/col needs >= this many ink px to count (ignores specks)
PAD = 40         # uniform white margin kept around the crop


def _need(tool):
    if shutil.which(tool) is None:
        raise SystemExit(f"'{tool}' not found on PATH -- cannot render (install texlive / poppler-utils)")


def render():
    _need("pdflatex"); _need("pdftoppm")
    with tempfile.TemporaryDirectory() as td:
        tex = os.path.join(td, "algo.tex")
        shutil.copyfile(SRC, tex)
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "algo.tex"],
            cwd=td, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        pdf = os.path.join(td, "algo.pdf")
        if r.returncode != 0 or not os.path.exists(pdf):
            tail = "\n".join(r.stdout.splitlines()[-30:])
            raise SystemExit(f"pdflatex failed (exit {r.returncode}):\n{tail}")
        subprocess.run(["pdftoppm", "-png", "-r", str(DPI), pdf,
                        os.path.join(td, "page")], check=True)
        return Image.open(os.path.join(td, "page-1.png")).convert("RGB").copy()


def main():
    im = render()
    dark = np.asarray(im.convert("L")) < THRESH
    ys = np.where(dark.sum(axis=1) >= MIN_RUN)[0]
    xs = np.where(dark.sum(axis=0) >= MIN_RUN)[0]
    if ys.size == 0 or xs.size == 0:
        raise SystemExit("no content found -- check THRESH / the source compiles")
    W, H = im.size
    box = (max(0, xs.min() - PAD), max(0, ys.min() - PAD),
           min(W, xs.max() + 1 + PAD), min(H, ys.max() + 1 + PAD))
    crop = im.crop(box)
    crop.save(OUT, optimize=True)
    print(f"render {W}x{H} @ {DPI}dpi -> crop {crop.size}  box={box}")
    print(f"saved {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB)")
    print("remember: re-upload algo_stride.png to HackMD ([algo_stride] in main_zh.md)")


if __name__ == "__main__":
    main()
