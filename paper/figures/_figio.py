"""Shared figure output for the manuscript figures.

Every `make_*_fig.py` under `paper/figures/` writes through `save_figure()` here
instead of calling `fig.savefig` directly, so raster resolution and the set of
formats are decided in exactly one place.

Import it from a sibling directory with:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _figio import DPI, FIG_FORMATS, save_figure

which resolves the same way wherever the script is started from — the repo root,
its own directory, or an editor's run button.

Figures are named after their thesis caption ("Figure 13. Training reward on
GÉANT") so the file drops into the document without renaming. Keep those names
in step with the List of Figures.
"""
import os

import matplotlib as mpl

DPI = 600                                    # raster DPI (png/jpg); svg/pdf are vector
FIG_FORMATS = ("png", "jpg", "svg", "pdf")   # every figure is written in all four

# Stamped into every SVG in place of the time it was written. When a figure was
# regenerated is not information about the figure, and letting it vary means
# every vector file changes whenever anyone re-runs a generator -- which is how a
# machine that had only rebuilt the figures ended up unable to git pull.
FIXED_DATE = "2000-01-01T00:00:00"


def save_figure(fig, out_dir, basename, bbox_inches=None, formats=None, dpi=None):
    """Write one figure under `basename` once per format in FIG_FORMATS.

    `basename` carries NO extension — it is the thesis caption. PNG/JPG are
    raster and honour DPI. SVG/PDF are vector, so DPI only affects rasterised
    elements inside them. JPG has no alpha channel, so it is written on a white
    background.

    `formats` and `dpi` override the module defaults for one call; presentation
    figures that want a single lightweight PNG pass them.
    """
    exts = formats if formats is not None else FIG_FORMATS
    d = dpi if dpi is not None else DPI
    written = []
    for ext in exts:
        out = os.path.join(out_dir, f"{basename}.{ext}")
        kw = {}
        if bbox_inches:
            kw["bbox_inches"] = bbox_inches
        if ext == "png":
            kw.update(dpi=d, pil_kwargs={"optimize": True})
        elif ext == "jpg":
            kw.update(dpi=d, facecolor="white",
                      pil_kwargs={"quality": 95, "optimize": True})
        elif ext == "pdf":
            # Drops /CreationDate. Verified: without it two runs a second apart
            # differ, with it they are byte-identical.
            kw["metadata"] = {"CreationDate": None}
        elif ext == "svg":
            # SVG needs both halves, and None is not enough here the way it is
            # for PDF -- passing None leaves matplotlib writing today's date into
            # <dc:date>, so the date has to be a fixed string. The hashsalt pins
            # the element ids, which are otherwise salted per process and rewrite
            # every clip-path reference in the file.
            kw["metadata"] = {"Date": FIXED_DATE}
        with mpl.rc_context({"svg.hashsalt": basename}):
            fig.savefig(out, **kw)
        written.append(out)
        print(f"saved {out}  ({os.path.getsize(out) / 1024:.0f} KB)")
    return written
