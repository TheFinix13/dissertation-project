#!/usr/bin/env python3
"""Rebuild Ch.4 figures at high resolution and refresh them in Full report - 4.docx."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "latex/final_dissertation/figs"
DOC = Path(
    "/Users/the1finix/Documents/University of Surrey/Feb 2026 Courses/"
    "Surrey Courses/Final Dissertation/Dissertation Drafts/Full report - 4.docx"
)
OUT = DOC.parent / "Full report - 4 - sharp figures.docx"

DPI = 400
WIDTH_CM = 16.0


def build_png(stem: str) -> Path:
    tex = FIGS / f"{stem}.tex"
    pdf = FIGS / f"{stem}.pdf"
    png = FIGS / f"{stem}.png"
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex.name],
        cwd=FIGS, check=True, capture_output=True,
    )
    # Prefer pdftoppm for crisp rasterisation; fall back to sips at higher density.
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(DPI), "-singlefile", str(pdf), str(FIGS / stem)],
            check=True, capture_output=True,
        )
        produced = FIGS / f"{stem}.png"
        if produced != png and produced.exists():
            produced.replace(png)
    except (FileNotFoundError, subprocess.CalledProcessError):
        subprocess.run(
            ["sips", "-s", "format", "png", str(pdf), "--out", str(png)],
            check=True, capture_output=True,
        )
    if not png.exists():
        raise FileNotFoundError(f"failed to build {png}")
    return png


def paragraph_has_drawing(p) -> bool:
    return bool(p._element.findall(".//" + qn("w:drawing")))


def replace_figure_before_caption(doc: Document, caption_prefix: str, png: Path) -> bool:
    cap_idx = next(
        (i for i, p in enumerate(doc.paragraphs) if p.text.strip().startswith(caption_prefix)),
        None,
    )
    if cap_idx is None:
        return False
    img_idx = None
    for i in range(cap_idx - 1, max(cap_idx - 4, -1), -1):
        if paragraph_has_drawing(doc.paragraphs[i]):
            img_idx = i
            break
    if img_idx is None:
        return False
    p = doc.paragraphs[img_idx]
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(png), width=Cm(WIDTH_CM))
    return True


def main() -> None:
    pipeline_png = build_png("system_pipeline")
    chrono_png = build_png("chrono_split")
    print(f"built {pipeline_png} ({pipeline_png.stat().st_size // 1024} KB)")
    print(f"built {chrono_png} ({chrono_png.stat().st_size // 1024} KB)")

    doc = Document(str(DOC))
    ok1 = replace_figure_before_caption(doc, "Figure 4.1:", pipeline_png)
    ok2 = replace_figure_before_caption(doc, "Figure 4.2:", chrono_png)
    if not ok1:
        raise RuntimeError("could not find Figure 4.1 image paragraph")
    if not ok2:
        raise RuntimeError("could not find Figure 4.2 image paragraph")

    doc.save(str(OUT))
    try:
        shutil.copy2(OUT, DOC)
        print(f"updated {DOC}")
    except Exception as exc:
        print(f"saved to {OUT} ({exc})")


if __name__ == "__main__":
    main()
