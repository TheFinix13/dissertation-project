#!/usr/bin/env python3
"""Reorder Chapter 4 of Full report - 4.docx into a reader-friendly arrangement.

Output goes to a new file: Full report - 4 - reordered.docx. The source is not
modified. Section renumbering is applied in place; figure numbers are swapped so
Figure 4.1 is the system pipeline (overview) and Figure 4.2 is the chronological
split (data pipeline).

Also empties the Benchmark Implementation section: the title '4.6 Benchmark
Implementation' remains as a placeholder so the section slot is visible; the
prose is removed pending user verification.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

SRC = Path(
    "/Users/the1finix/Documents/University of Surrey/Feb 2026 Courses/"
    "Surrey Courses/Final Dissertation/Dissertation Drafts/Full report - 4.docx"
)
OUT = SRC.parent / "Full report - 4 - reordered.docx"


# ---------------------------------------------------------------- ranges
# Each tuple is (first_index, last_index) inclusive, referencing paragraph
# indices in the source document. Determined by inspection.

BLOCKS = {
    "title":        (61, 63),   # Chapter 4 heading + lead paragraph
    "4.1":          (64, 67),   # Introduction
    "4.2_hdr":      (68, 68),   # "4.2 Data Pipeline" heading
    "4.2.1":        (69, 74),
    "4.2.2":        (75, 85),
    "4.2.3":        (86, 98),   # includes Fig 4.1 image + caption
    "4.3_hdr":      (99, 100),  # "4.3 TVT" heading + intro
    "4.3.1":        (101, 105),
    "4.3.2":        (106, 113),
    "4.3.3":        (114, 120),
    "4.4_hdr":      (121, 124),
    "4.4.1":        (125, 138),
    "4.4.2":        (139, 146),
    "4.5_hdr":      (147, 150),
    "4.5.1":        (151, 155),
    "4.5.2":        (156, 164),
    "4.5.3":        (165, 170),
    "4.5.4":        (171, 175),
    "4.6_hdr":      (176, 176),
    "4.6.1":        (177, 183),
    "4.6.2":        (184, 194),
    "4.7_pipeline": (195, 206),  # System pipeline overview + Fig 4.2 image + caption
    "trailing":     (207, 216),  # blanks + old orphan benchmark section
}

# ---------------------------------------------------------------- helpers


def replace_run_text(p, old: str, new: str) -> bool:
    """Replace the first run whose text contains `old` with `new`. Returns True
    if a replacement was made."""
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            return True
    return False


def rewrite_heading(p, new_text: str) -> None:
    """Replace the entire text of a heading paragraph, preserving the first
    run's formatting."""
    if not p.runs:
        p.add_run(new_text)
        return
    p.runs[0].text = new_text
    for r in p.runs[1:]:
        r.text = ""


def clone_heading_paragraph(template_p, new_text: str):
    """Deep-copy a heading paragraph's XML and set its text to `new_text`."""
    new_p = deepcopy(template_p._element)
    # Wipe existing text runs' content, then set the first run's text.
    runs = new_p.findall(qn("w:r"))
    if not runs:
        # Build a minimal run with bold
        r = OxmlElement("w:r")
        rPr = OxmlElement("w:rPr")
        b = OxmlElement("w:b")
        rPr.append(b)
        r.append(rPr)
        t = OxmlElement("w:t")
        t.text = new_text
        r.append(t)
        new_p.append(r)
    else:
        first = True
        for r in runs:
            for t in r.findall(qn("w:t")):
                if first:
                    t.text = new_text
                    first = False
                else:
                    t.text = ""
    return new_p


# ---------------------------------------------------------------- main


def main() -> None:
    doc = Document(str(SRC))
    paras = doc.paragraphs

    # --- 1. Renumber section headings in place -----------------------------
    heading_renames = {
        68:  "4.3 Data Pipeline",
        69:  "4.3.1 Market Data Preparation",
        75:  "4.3.2 Feature Construction & Continuous Calculation",
        86:  "4.3.3 Dataset Split & Monthly Episodes",
        99:  "4.4 Training, Validation and Testing Procedure",
        101: "4.4.1 Training",
        106: "4.4.2 Validation",
        114: "4.4.3 Testing",
        121: "4.5 Experiment Configuration",
        125: "4.5.1 Configuration Parameters",
        139: "4.5.2 Algorithm Configuration",
        147: "4.7 Implementation Verifications",
        151: "4.7.1 Environment and Portfolio Tests",
        156: "4.7.2 Feature and Data Tests",
        165: "4.7.3 Reward and Risk Tests",
        171: "4.7.4 Action-Masking Tests",
        176: "4.8 Implementation Corrections",
        177: "4.8.1 Problems Identified During Development",
        184: "4.8.2 Correction and Re-verification",
        195: "4.2 System Overview",
    }
    for idx, new_text in heading_renames.items():
        rewrite_heading(paras[idx], new_text)

    # --- 2. Swap figure numbers ------------------------------------------
    # Old Fig 4.1 (chrono split, in 4.2.3 -> new 4.3.3) becomes Fig 4.2.
    # Old Fig 4.2 (system pipeline, in old 4.7 -> new 4.2) becomes Fig 4.1.
    replace_run_text(paras[89], "1", "2")   # "Figure 4.1 shows the split." -> "4.2"
    replace_run_text(paras[91], "1", "2")   # Figure 4.1 caption -> Figure 4.2
    replace_run_text(paras[204], "2", "1")  # "Figure 4.2 summarises the flow." -> "4.1"
    replace_run_text(paras[206], "2", "1")  # Figure 4.2 caption -> Figure 4.1

    # --- 3. Prepare the new benchmark heading (empty section) -------------
    benchmark_heading = clone_heading_paragraph(paras[68], "4.6 Benchmark Implementation")

    # --- 4. Collect XML element references for every block ---------------
    block_elements = {
        key: [paras[i]._element for i in range(a, b + 1)]
        for key, (a, b) in BLOCKS.items()
    }
    all_chapter_elems = [e for key in BLOCKS for e in block_elements[key]]

    # --- 5. Detach every Chapter 4 element from body ---------------------
    body = doc.element.body
    # The sectPr may or may not sit at the end of body; record it if present.
    sectPr = body.find(qn("w:sectPr"))
    insertion_anchor = None
    # Insert after the paragraph immediately before Chapter 4 (paragraph 60).
    insertion_anchor = paras[60]._element

    for elem in all_chapter_elems:
        body.remove(elem)

    # --- 6. Rebuild in the new order --------------------------------------
    NEW_ORDER = [
        "title",
        "4.1",
        # 4.2 System Overview (was old 4.7 pipeline)
        "4.7_pipeline",
        # 4.3 Data Pipeline (was old 4.2)
        "4.2_hdr",
        "4.2.1",
        "4.2.2",
        "4.2.3",
        # 4.4 Training / Validation / Testing (was old 4.3)
        "4.3_hdr",
        "4.3.1",
        "4.3.2",
        "4.3.3",
        # 4.5 Experiment Configuration (was old 4.4)
        "4.4_hdr",
        "4.4.1",
        "4.4.2",
        # 4.6 Benchmark Implementation (title only)
        # (inserted below as a single new element)
        # 4.7 Implementation Verifications (was old 4.5)
        "4.5_hdr",
        "4.5.1",
        "4.5.2",
        "4.5.3",
        "4.5.4",
        # 4.8 Implementation Corrections (was old 4.6)
        "4.6_hdr",
        "4.6.1",
        "4.6.2",
    ]

    # Insert elements sequentially after the anchor.
    cursor = insertion_anchor
    for key in NEW_ORDER:
        for elem in block_elements[key]:
            cursor.addnext(elem)
            cursor = elem
        # After the last block of 4.5.2 (Algorithm Configuration), insert the
        # empty benchmark heading before starting section 4.7 verifications.
        if key == "4.4.2":
            cursor.addnext(benchmark_heading)
            cursor = benchmark_heading

    # Sanity: sectPr should still be at the end of body; if we accidentally
    # displaced it, python-docx will restore serialisation on save.
    doc.save(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
