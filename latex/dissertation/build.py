#!/usr/bin/env python3
r"""Post-process pandoc-converted chapter fragments into a dissertation build.

Reads:
    ch3_part1.tex  (from Full Report - 3 .docx)
    ch34_rest.tex  (from Full report - 4.docx: Ch3 tail + Ch4 + Ch5 5.1-5.6)
    ch5_part2.tex  (from Full Report - 5.docx: Ch5 5.7-5.10 + Ch6)
    ch1_new.tex    (from Full report 1.docx: Ch1 + new 6.4.2 + rewritten
                    5.2 and 4.4.3 as trailing blocks)

Writes:
    ch1.tex   (all of Chapter 1, cleaned)
    ch3.tex   (all of Chapter 3, cleaned)
    ch4.tex   (all of Chapter 4, cleaned, with the author's 4.4.3)
    ch5.tex   (all of Chapter 5, cleaned, with the author's 5.2)
    ch6.tex   (all of Chapter 6, with 6.4.2 spliced in before Fixed Trade Size)
    main.tex  (professional dissertation preamble, chapter includes, TOC)

The heading fixup converts pandoc's `\textbf{N.M[.P[.Q]] Title}` markers into
proper \section / \subsection / \subsubsection, and turns the two-line
`Chapter N` + Title pair at the top of each fragment into a \chapter{Title}.
Headings that Word stored as plain (non-bold) standalone lines are converted
too.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PART1 = HERE / "ch3_part1.tex"
REST = HERE / "ch34_rest.tex"
PART2 = HERE / "ch5_part2.tex"
PARTC1 = HERE / "ch1_new.tex"
OUT_CH1 = HERE / "ch1.tex"
OUT_CH3 = HERE / "ch3.tex"
OUT_CH4 = HERE / "ch4.tex"
OUT_CH5 = HERE / "ch5.tex"
OUT_CH6 = HERE / "ch6.tex"
OUT_MAIN = HERE / "main.tex"

# The author writes the Chapter 6 banner as one bold line, e.g.
# \textbf{Chapter 6 -- Conclusion and Future Work.}
CH6_BANNER_RE = re.compile(
    r"^\\textbf\{Chapter\s+6\s*(?:--|-|—)?\s*(?P<title>[^{}]*?)[.\s]*\}\s*$",
    re.MULTILINE,
)


# ----------------------------------------------------------------- helpers


HEAD_RE = re.compile(
    r"^\\textbf\{(?P<num>\d+(?:\.\d+){1,3})[\s.]*"
    r"(?P<rest>[^{}]*(?:\{[^{}]*\}[^{}]*)*?)\}"  # allow one level of nested braces
    r"(?P<tail>.*)$"
)
# Match pandoc's "\textbf{3.2.1} \textbf{Episode definition} \(\mathbf{(\tau)}\)"
# style where the number is one bold block and the rest is another.
SPLIT_HEAD_RE = re.compile(
    r"^\\textbf\{(?P<num>\d+(?:\.\d+){1,3})\}\s*(?P<tail>.+)$"
)
# Word sometimes stores a heading as a plain paragraph: "5.7 Effect of the
# Risk-Aware Reward" with no bold markup. Only treat a line as such a heading
# when it stands alone as its own paragraph and looks like a title.
PLAIN_HEAD_RE = re.compile(
    r"^(?P<num>\d+\.\d+(?:\.\d+){0,2})\s+(?P<title>[A-Z][^\\{}]{2,80})$"
)


def _level_from_number(num: str) -> str:
    """3.1 -> section, 3.1.1 -> subsection, 3.1.1.1 -> subsubsection."""
    depth = num.count(".")
    if depth == 1:
        return "section"
    if depth == 2:
        return "subsection"
    if depth == 3:
        return "subsubsection"
    return "paragraph"


def _strip_bold_wrappers(text: str) -> str:
    """Collapse `\textbf{A} \textbf{B}` -> `A B` for heading titles."""
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", text)
    return text.strip()


def _clean_heading_title(text: str) -> str:
    """Post-process the visible portion of a heading."""
    text = _strip_bold_wrappers(text)
    # \mathbf in a heading asks unicode-math for bold glyphs from the *text*
    # bold font, which lacks greek; headings are bold anyway, so drop it.
    text = text.replace(r"\mathbf{(\tau)}", r"(\tau)")
    # Wrap any inline math in \texorpdfstring so bookmarks stay valid.
    def _protect_math(match):
        inner = match.group(1)
        # Best-effort plain rendering by dropping backslashed commands.
        plain = re.sub(r"\\[a-zA-Z]+\s*", "", inner)
        plain = plain.replace("{", "").replace("}", "").strip()
        return r"\texorpdfstring{\(" + inner + r"\)}{" + (plain or "?") + "}"

    text = re.sub(r"\\\((.+?)\\\)", _protect_math, text)
    # Trim trailing colon/period.
    text = re.sub(r"[\s:.,]+$", "", text)
    # Sanitise stray backslashes at end.
    text = text.strip()
    return text


def join_wrapped_bold_headings(source: str) -> str:
    r"""Merge hard-wrapped `\textbf{5.8.1 ...` heading paragraphs onto one line.

    Pandoc wraps long lines, so a bold heading can end up split across two
    or three lines with the closing brace on the last. The per-line heading
    conversion needs them joined first.
    """
    lines = source.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"^\\textbf\{\d+(?:\.\d+){1,3}\s", line.strip()):
            merged = line.strip()
            while merged.count("{") > merged.count("}") and i + 1 < len(lines):
                i += 1
                merged += " " + lines[i].strip()
            out.append(merged)
        else:
            out.append(line)
        i += 1
    return "\n".join(out) + "\n"


def convert_headings(source: str) -> str:
    """Convert \\textbf{N.M[.P[.Q]] Title} lines to proper section commands.

    Only applies to lines that BEGIN with a matching \\textbf and where the
    number pattern indicates a heading. Prose in bold is untouched. Plain
    standalone "N.M Title" paragraph lines are converted as well.
    """
    out: list[str] = []
    lines = source.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.rstrip()
        # Bold table-row labels (`\textbf{0.05} & ... \\`) must never be
        # mistaken for headings: any cell separator or row terminator on
        # the line disqualifies it.
        if " & " in stripped or stripped.endswith("\\\\"):
            out.append(line)
            continue
        m = HEAD_RE.match(stripped)
        if not m:
            m2 = SPLIT_HEAD_RE.match(stripped)
            if m2:
                num = m2.group("num")
                title = _clean_heading_title(m2.group("tail"))
                out.append(f"\\{_level_from_number(num)}{{{title}}}")
                continue
            m3 = PLAIN_HEAD_RE.match(stripped.strip())
            prev_blank = idx == 0 or not lines[idx - 1].strip()
            next_blank = idx == len(lines) - 1 or not lines[idx + 1].strip()
            if m3 and prev_blank and next_blank:
                num = m3.group("num")
                title = _clean_heading_title(m3.group("title"))
                out.append(f"\\{_level_from_number(num)}{{{title}}}")
                continue
            out.append(line)
            continue
        num = m.group("num")
        rest = m.group("rest").strip()
        tail = m.group("tail").strip()
        title_parts = [rest]
        if tail:
            title_parts.append(_strip_bold_wrappers(tail))
        title = _clean_heading_title(" ".join(t for t in title_parts if t))
        out.append(f"\\{_level_from_number(num)}{{{title}}}")
    return "\n".join(out) + "\n"


def strip_chapter_banner(source: str, chapter_title: str | None = None):
    """Remove the pandoc-produced 'Chapter N' + subtitle preamble.

    Returns (body_without_banner, detected_title). If a chapter_title is passed
    in, uses it; otherwise inspects the first two \\textbf lines.
    """
    lines = source.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    m = re.match(r"^\\textbf\{Chapter\s+\d+\}\s*$", lines[i].strip())
    if not m:
        return source, chapter_title
    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    title = chapter_title
    if i < len(lines):
        tm = re.match(r"^\\textbf\{(?P<t>.+)\}\s*$", lines[i].strip())
        if tm:
            title = _clean_heading_title(tm.group("t"))
            i += 1
    remainder = "\n".join(lines[i:]).lstrip("\n")
    return remainder, title


def split_at_chapter(source: str, chapter_marker: str):
    """Split the combined fragment at a chapter banner."""
    idx = source.find(chapter_marker)
    if idx == -1:
        raise RuntimeError(f"{chapter_marker} banner not found in fragment")
    return source[:idx].rstrip() + "\n", source[idx:]


CAPTION_START_RE = re.compile(r"^(?:\\emph\{)?Figure\s+\d+(?:\.\d+)?\s*[:.]")
CAPTION_PREFIX_RE = re.compile(r"^Figure\s+\d+(?:\.\d+)?\s*[:.]?\s*")


def shorten_lof_caption(caption: str) -> str:
    """Compact List-of-Figures title; the full caption stays under the figure.

    Rules follow common dissertation practice and the author's guidance:
    - Left:/Right: panels keep the title plus both panel clauses, drop later commentary.
    - ``Title: elaboration`` uses only the title before the first colon.
    - Otherwise keep the first sentence only.
    """
    text = re.sub(r"\\textbf\{\.\}\s*$", ".", caption.strip()).strip()
    if re.search(r"\bLeft:", text) and re.search(r"\bRight:", text):
        parts = re.split(r"(?<=\.)\s+", text)
        kept = []
        for part in parts:
            kept.append(part)
            if re.search(r"\bRight:", part):
                break
        return " ".join(kept).strip()
    colon = text.find(":")
    period = text.find(".")
    if colon != -1 and (period == -1 or colon < period):
        before = text[:colon].strip()
        after = text[colon + 1 :].strip()
        if 15 <= len(before) <= 90 and len(after) > 15:
            return before
    m = re.match(r"^(.+?\.)(?:\s+|$)", text)
    if m and len(m.group(1)) < len(text) - 5:
        return m.group(1).strip()
    return text


def wrap_figures(source: str) -> str:
    r"""Wrap bare \includegraphics + following caption into a figure environment.

    Captions in the source are paragraphs like ``Figure 4.1: text...`` or
    ``\emph{Figure 4.1: text...}`` possibly spanning several lines. The
    numeric prefix is dropped so LaTeX's own figure numbering (which matches
    the author's numbers) provides the label. A short LoF title is supplied
    via ``\caption[short]{full}`` so the list stays readable.
    """
    out = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        gm = re.match(r"^\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}\s*$", line.strip())
        if gm:
            path = gm.group(1)
            # find the caption paragraph on the next non-blank line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            caption = None
            if j < len(lines) and CAPTION_START_RE.match(lines[j].strip()):
                para = []
                while j < len(lines) and lines[j].strip():
                    para.append(lines[j].strip())
                    j += 1
                caption = " ".join(para)
                if caption.startswith("\\emph{") and caption.endswith("}"):
                    caption = caption[len("\\emph{"):-1]
                caption = CAPTION_PREFIX_RE.sub("", caption).strip()
            # [!htb] rather than [H]: a pinned figure that does not fit
            # leaves the rest of the page empty, whereas a float moves to
            # the next page top and lets the text fill the gap.
            out.append("\\begin{figure}[!htb]")
            out.append("  \\centering")
            out.append(f"  \\includegraphics[width=0.9\\textwidth]{{{path}}}")
            if caption:
                short = shorten_lof_caption(caption)
                if short and short != caption:
                    out.append(f"  \\caption[{short}]{{{caption}}}")
                else:
                    out.append(f"  \\caption{{{caption}}}")
            out.append("\\end{figure}")
            out.append("")
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + "\n"


def fix_bold_math(source: str) -> str:
    r"""Route bold Greek through the math font.

    Word emits bold inline math like \(\mathbf{\lambda = 0.10}\). Under
    unicode-math, \mathbf asks the *text* bold font for Greek glyphs it
    does not have; \symbf uses STIX Two Math's bold alphabet instead.
    """
    return source.replace(r"\mathbf{\lambda", r"\symbf{\lambda")


ALGO_BOX_RE = re.compile(
    # Opening brace + longtable header, at most a few lines of column spec
    # and rules before the minipage. The minipage must OPEN with the
    # algorithm title, so ordinary tables can never match.
    r"\{\\def\\LTcaptype\{none\}[^\n]*\n"
    r"\\begin\{longtable\}[^\n]*\n"
    r"(?:[^\n]*\n){0,4}?"
    r"\\begin\{minipage\}\[b\]\{\\linewidth\}\\raggedright\n"
    r"(?P<content>\\textbf\{Algorithm\s+\d+.*?)"
    r"\\end\{minipage\}[^\n]*\n"
    r"(?:[^\n]*\n){0,6}?"
    r"\\end\{longtable\}\s*\n\}",
    re.DOTALL,
)


def unbox_algorithms(source: str) -> str:
    r"""Turn the pandoc algorithm tables into breakable ruled blocks.

    Pandoc renders each algorithm as a single-cell longtable whose whole
    body sits in the (unbreakable) header. A half-page block that does not
    fit is pushed to the next page, leaving a large gap. A plain ruled
    block keeps the boxed look but can break across pages.
    """
    def repl(m: re.Match) -> str:
        content = m.group("content").rstrip()
        # Thin rule between the title line and the numbered steps, matching
        # the original head/body separator.
        content = content.replace(
            "\n\n\\begin{quote}",
            "\n\\par\\vspace{4pt}\\hrule height 0.4pt\\vspace{2pt}\n"
            "\\begin{quote}",
            1,
        )
        return ("\\begin{algobox}\n" + content + "\n\\end{algobox}")

    return ALGO_BOX_RE.sub(repl, source)


TABLE_CAP_RE = re.compile(
    r"^(?:\\emph\{|\\textbf\{)Table\s+(?P<num>\d+\.\d+[a-z]?)\s*[:.]")


def add_table_lot_entries(source: str) -> str:
    r"""Register the author's "Table N.M: ..." caption paragraphs in the LoT.

    The Word documents number tables manually and write the caption as an
    \emph or \textbf paragraph, so LaTeX's \listoftables sees nothing. This
    inserts an \addcontentsline before each caption paragraph, using the
    author's own number. Duplicate numbers (e.g. Table 3.3b appears as both
    a bold title and an italic caption) keep only the first occurrence.
    """
    lines = source.splitlines()
    out: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        m = TABLE_CAP_RE.match(lines[i].strip())
        if m and m.group("num") not in seen:
            num = m.group("num")
            seen.add(num)
            para = []
            j = i
            while j < len(lines) and lines[j].strip():
                para.append(lines[j].strip())
                j += 1
            text = " ".join(para)
            # Neutralise formatting wrappers into plain groups (keeps braces
            # balanced even when the caption contains math), then drop the
            # "Table N.M:" prefix inside the first group.
            text = text.replace(r"\emph{", "{").replace(r"\textbf{", "{")
            text = re.sub(
                r"\{\s*Table\s+[\d.]+[a-z]?\s*[:.]\s*", "{", text, count=1)
            out.append(
                r"\addcontentsline{lot}{table}{\protect\numberline{"
                + num + r"}" + text + "}")
        out.append(lines[i])
        i += 1
    return "\n".join(out) + "\n"


def add_row_separators(source: str) -> str:
    r"""Insert \rowsep hairlines between longtable body rows.

    Pandoc emits longtables with rules only at the head and foot, so rows
    with wrapped cells blur together. The body starts after \endlastfoot;
    every line ending in `\\` there terminates one logical row. The last
    row is skipped because \bottomrule already follows it. Single-row
    tables (e.g. the algorithm boxes) have no body rows and are untouched.
    """
    out: list[str] = []
    lines = source.splitlines()
    in_body = False
    pending_sep = False
    minipage_depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped == r"\endlastfoot":
            in_body = True
            pending_sep = False
            minipage_depth = 0
            out.append(line)
            continue
        if stripped == r"\end{longtable}":
            in_body = False
            pending_sep = False  # never rule before \bottomrule
            out.append(line)
            continue
        if in_body:
            if pending_sep and stripped:
                out.append(r"\rowsep")
                pending_sep = False
            minipage_depth += stripped.count(r"\begin{minipage}")
            minipage_depth -= stripped.count(r"\end{minipage}")
            # `\\` inside a minipage is a cell-internal line break, not a
            # row terminator.
            if minipage_depth == 0 and stripped.endswith(r"\\"):
                pending_sep = True
        out.append(line)
    return "\n".join(out) + "\n"


# --------------------------------------------------- author-update splicing
# Full report 1.docx carries, after Chapter 1 itself, three trailing blocks
# the author rewrote: 6.4.2 (new limitation), 5.2 (protocols) and 4.4.3
# (benchmark implementation). Each starts with its own bold heading.

CH1_SPLIT_MARKERS = (
    r"\textbf{6.4.2 Monthly Episode Structure}",
    r"\textbf{5.2 Experiment Protocols and Evaluation Criteria}",
    r"\textbf{4.4.3 Benchmark Implementation}",
)


def split_ch1_fragment(source: str):
    """Return (ch1_body, sec642, sec52, sec443) from the Report-1 fragment."""
    positions = []
    for marker in CH1_SPLIT_MARKERS:
        idx = source.find(marker)
        if idx == -1:
            raise RuntimeError(f"marker not found in ch1 fragment: {marker!r}")
        positions.append(idx)
    if positions != sorted(positions):
        raise RuntimeError("ch1 fragment blocks out of expected order")
    ch1_body = source[:positions[0]]
    sec642 = source[positions[0]:positions[1]]
    sec52 = source[positions[1]:positions[2]]
    sec443 = source[positions[2]:]
    # The document ends with front-matter blocks (Statement of Originality,
    # Acknowledgements, Abstract) that live statically in main_full.tex, not
    # in any chapter. Cut 4.4.3 short so they don't leak into Chapter 4.
    fm = sec443.find(r"\textbf{Statement of Originality}")
    if fm != -1:
        sec443 = sec443[:fm]
    sec443 = re.sub(
        r"\\textbf\{\\hfill\\break\s*\}\s*$", "", sec443.rstrip())
    # Strip the single-line chapter banner and the trailing \hfill\break junk
    # Word leaves between the chapter and the extra blocks.
    ch1_body = re.sub(
        r"^\\textbf\{Chapter\s+1\s*(?:--|-|—)?\s*[^{}]*\}\s*\n", "", ch1_body)
    ch1_body = re.sub(r"\\textbf\{\\hfill\\break\s*\}\s*$", "", ch1_body.rstrip())
    # 1.6 writes "state-representation" once; the compound blocks TeX's
    # line-breaking and overflows the margin. The rest of the chapter uses
    # the unhyphenated form, so normalise this occurrence to match.
    ch1_body = ch1_body.replace(
        "state-representation and", "state representation and")
    return ch1_body.rstrip() + "\n", sec642, sec52, sec443


def replace_span(source: str, start_marker: str, end_marker: str,
                 replacement: str) -> str:
    """Replace source[start_marker:end_marker) with the replacement block."""
    i = source.find(start_marker)
    if i == -1:
        raise RuntimeError(f"start marker not found: {start_marker!r}")
    j = source.find(end_marker, i)
    if j == -1:
        raise RuntimeError(f"end marker not found: {end_marker!r}")
    return source[:i] + replacement.rstrip() + "\n\n" + source[j:]


def insert_before(source: str, marker: str, block: str) -> str:
    """Insert a block immediately before the marker line."""
    i = source.find(marker)
    if i == -1:
        raise RuntimeError(f"marker not found: {marker!r}")
    return source[:i] + block.rstrip() + "\n\n" + source[i:]


# ------------------------------------------------------------------- main


# ---------------------------------------------------------------------------
# IEEE citation pass. The chapters come from the author's Word documents,
# which carry no citations, so the keys from references.bib are attached to
# specific claim points here. Each anchor must appear exactly once in the
# generated chapter; a changed Word document prints a warning rather than
# breaking the build.
CITATIONS = {
    "ch1": [
        ("increasingly applied to financial trading.",
         "increasingly applied to financial "
         "trading~\\cite{moody2001, deng2017, fischer2018}."),
        ("reason reinforcement learning (RL) is appropriate",
         "reason reinforcement learning (RL)~\\cite{sutton2018} "
         "is appropriate"),
        ("algorithms --- REINFORCE, Deep Q-learning (DQN), and Proximal Policy",
         "algorithms --- REINFORCE~\\cite{williams1992}, Deep Q-learning "
         "(DQN)~\\cite{mnih2015}, and Proximal Policy"),
        ("Optimization (PPO) ---",
         "Optimization (PPO)~\\cite{schulman2017ppo} ---"),
    ],
    "ch3": [
        ("Markov Decision Process (MDP), i.e",
         "Markov Decision Process (MDP)~\\cite{bellman1957, sutton2018}, i.e"),
        ("\\textbf{REINFORCE}---the Monte-Carlo policy gradient, which",
         "\\textbf{REINFORCE}~\\cite{williams1992}---the Monte-Carlo "
         "policy gradient, which"),
        ("\\textbf{Deep Q-learning}, which learns",
         "\\textbf{Deep Q-learning}~\\cite{watkins1992, mnih2015}, "
         "which learns"),
        ("Proximal Policy Optimisation (PPO) is used as the third",
         "Proximal Policy Optimisation (PPO)~\\cite{schulman2017ppo} "
         "is used as the third"),
        ("\\textbf{MaskablePPO} from sb3-contrib.",
         "\\textbf{MaskablePPO} from sb3-contrib~\\cite{raffin2021sb3}."),
        ("the \\textbf{Adam optimiser} with a learning rate",
         "the \\textbf{Adam optimiser}~\\cite{kingma2015adam} "
         "with a learning rate"),
        ("the Bellman optimality equation",
         "the Bellman optimality equation~\\cite{bellman1957}"),
        ("an \\textbf{experience replay buffer}.",
         "an \\textbf{experience replay buffer}~\\cite{lin1992, mnih2015}."),
        ("\\textbf{target network}:",
         "\\textbf{target network}~\\cite{mnih2015}:"),
        ("described in Section 3.4 also applies to PPO.",
         "described in Section 3.4 also applies to "
         "PPO~\\cite{huang2022masking}."),
    ],
    "ch4": [
        ("PyTorch. PPO was implemented using the MaskablePPO implementation",
         "PyTorch~\\cite{paszke2019}. PPO was implemented using the "
         "MaskablePPO implementation"),
        ("sb3-contrib. The Gymnasium interface simulates",
         "sb3-contrib~\\cite{raffin2021sb3}. The Gymnasium "
         "interface~\\cite{towers2024gymnasium} simulates"),
        ("Maintaining a chronological split is important for financial data.",
         "Maintaining a chronological split is important for financial "
         "data~\\cite{lopezdeprado2018}."),
        ("variation between runs to be reported.",
         "variation between runs to be reported~\\cite{henderson2018}."),
    ],
    "ch5": [
        ("and the monthly Sharpe ratio, a financial metric",
         "and the monthly Sharpe ratio~\\cite{sharpe1994}, "
         "a financial metric"),
        ("average alone may hide.",
         "average alone may hide~\\cite{henderson2018}."),
        ("To account for randomness in reinforcement learning, six random",
         "To account for randomness in reinforcement "
         "learning~\\cite{henderson2018}, six random"),
        ("returns are below the buy-and-hold benchmark.",
         "returns are below the buy-and-hold benchmark. This mirrors the "
         "fee response reported by Th\\'eate and Ernst, whose agent also "
         "traded less as the assumed costs rose~\\cite{theate2021}."),
        ("difficult benchmark to improve upon.",
         "difficult benchmark to improve upon, which is consistent with "
         "the market-efficiency expectation set out in "
         "Chapter 2~\\cite{fama1970}."),
    ],
    "ch6": [
        ("improve the reliability of comparisons",
         "improve the reliability of comparisons~\\cite{henderson2018}"),
        ("Sharpe-based or drawdown-relative objective.",
         "Sharpe-based or drawdown-relative "
         "objective~\\cite{moody2001}."),
    ],
}


def insert_citations(source: str, name: str) -> str:
    for old, new in CITATIONS.get(name, []):
        n = source.count(old)
        if n != 1:
            print(f"WARNING: citation anchor found {n}x in {name}: "
                  f"{old[:60]!r}")
            continue
        source = source.replace(old, new)
    return source


def main() -> None:
    # -----------------------------------------------------------------
    # FROZEN (Sep 2026): the chapter .tex files were hand-trimmed for the
    # 80-page target after the Word docs were finalised. Re-running this
    # script would overwrite those edits. Pass --force only if you really
    # want to regenerate the chapters from the docx fragments again.
    # -----------------------------------------------------------------
    if "--force" not in sys.argv:
        raise SystemExit(
            "build.py is frozen: ch1/ch3/ch4/ch5/ch6.tex now contain manual "
            "page-reduction edits that this script would destroy.\n"
            "Compile directly with:  xelatex main_full && bibtex main_full "
            "&& xelatex main_full && xelatex main_full\n"
            "Use 'python3 build.py --force' to regenerate anyway.")

    part1 = PART1.read_text()
    rest = REST.read_text()
    part2 = PART2.read_text()
    partc1 = PARTC1.read_text()

    ch1_body, sec642, sec52, sec443 = split_ch1_fragment(partc1)

    ch3_tail, ch4_and_5 = split_at_chapter(rest, "\\textbf{Chapter 4}")
    ch4_frag, ch5_frag = split_at_chapter(ch4_and_5, "\\textbf{Chapter 5}")

    # Split the author's Chapter 6 off the end of the Report-5 fragment.
    ch6_match = CH6_BANNER_RE.search(part2)
    if ch6_match:
        ch5b_part = part2[:ch6_match.start()].rstrip() + "\n"
        ch6_body = part2[ch6_match.end():].lstrip("\n")
        ch6_title = ch6_match.group("title").strip() or "Conclusion and Future Work"
    else:
        ch5b_part = part2
        ch6_body = None
        ch6_title = "Conclusion and Future Work"

    # Strip the "Chapter N" banner off each fragment that carries one.
    part1_body, ch3_title = strip_chapter_banner(part1)
    ch4_body, ch4_title = strip_chapter_banner(ch4_frag)
    ch5a_body, ch5_title = strip_chapter_banner(ch5_frag)

    ch3_combined = part1_body.rstrip() + "\n\n" + ch3_tail.lstrip()
    ch5_combined = ch5a_body.rstrip() + "\n\n" + ch5b_part.lstrip()

    # Splice the author's rewrites from Full report 1.docx.
    ch4_body = replace_span(
        ch4_body,
        r"\textbf{4.4.3 Benchmark Implementation}",
        r"\textbf{4.5 Implementation Verifications}",
        sec443,
    )
    ch5_combined = replace_span(
        ch5_combined,
        r"\textbf{5.2 Experiment Protocols and Evaluation Criteria}",
        r"\textbf{5.3 Fixed Trade Size Tuning}",
        sec52,
    )
    if ch6_body is not None:
        # The new limitation slots in before the author's old 6.4.2; LaTeX
        # renumbers everything below it automatically.
        ch6_body = insert_before(
            ch6_body, r"\textbf{6.4.2 Fixed Trade Size}", sec642)

    ch1_out = convert_headings(join_wrapped_bold_headings(ch1_body))
    ch1_out = fix_bold_math(ch1_out)
    ch1_out = insert_citations(ch1_out, "ch1")
    OUT_CH1.write_text(ch1_out)

    ch3_out = convert_headings(join_wrapped_bold_headings(ch3_combined))
    ch4_out = convert_headings(join_wrapped_bold_headings(ch4_body))
    ch5_out = convert_headings(join_wrapped_bold_headings(ch5_combined))

    ch3_out = wrap_figures(ch3_out)
    ch4_out = wrap_figures(ch4_out)
    ch5_out = wrap_figures(ch5_out)

    ch3_out = add_row_separators(ch3_out)
    ch4_out = add_row_separators(ch4_out)
    ch5_out = add_row_separators(ch5_out)

    ch3_out = add_table_lot_entries(ch3_out)
    ch4_out = add_table_lot_entries(ch4_out)
    ch5_out = add_table_lot_entries(ch5_out)

    ch3_out = unbox_algorithms(ch3_out)

    ch3_out = insert_citations(ch3_out, "ch3")
    ch4_out = insert_citations(ch4_out, "ch4")
    ch5_out = insert_citations(ch5_out, "ch5")

    OUT_CH3.write_text(ch3_out)
    OUT_CH4.write_text(ch4_out)
    OUT_CH5.write_text(ch5_out)

    if ch6_body is not None:
        ch6_out = convert_headings(join_wrapped_bold_headings(ch6_body))
        ch6_out = wrap_figures(ch6_out)
        ch6_out = add_row_separators(ch6_out)
        ch6_out = fix_bold_math(ch6_out)
        ch6_out = insert_citations(ch6_out, "ch6")
        OUT_CH6.write_text(ch6_out)
        print(f"wrote {OUT_CH6.name} ({ch6_title!r})")

    ch3_title = ch3_title or "Methodology and Mathematical Formulation"
    ch4_title = ch4_title or "System Design and Experiment Framework"
    ch5_title = ch5_title or "Experimental Results and Analysis"

    preamble = MAIN_TEMPLATE.format(
        ch3_title=ch3_title, ch4_title=ch4_title, ch5_title=ch5_title)
    OUT_MAIN.write_text(preamble)
    print(f"wrote {OUT_CH1.name}, {OUT_CH3.name}, {OUT_CH4.name}, "
          f"{OUT_CH5.name}, {OUT_MAIN.name}")


MAIN_TEMPLATE = r"""\documentclass[11pt,a4paper,openany]{{report}}

% --- geometry & typography ------------------------------------------------
\usepackage[a4paper,margin=1in,top=1.1in,bottom=1.2in]{{geometry}}
\usepackage{{microtype}}

% --- maths ----------------------------------------------------------------
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{mathtools}}

% --- fonts (xelatex) ------------------------------------------------------
\usepackage{{fontspec}}
\usepackage{{unicode-math}}
% Use STIX Two -- a Times-like family with a matching math font, installed
% by default on macOS. Falls back gracefully if unavailable.
% macOS ships STIX Two Text as a variable font, so the bold/italic faces
% must be requested explicitly or fontspec silently substitutes regular.
\IfFontExistsTF{{STIX Two Text}}{{%
  \setmainfont{{STIX Two Text}}[
    BoldFont={{STIX Two Text Bold}}]%
  \setmathfont{{STIX Two Math}}%
}}{{%
  % Leave xelatex defaults (Computer Modern) in place.
}}

% Word left some mathematical symbols as literal unicode characters in
% plain-text runs (mostly tables). STIX Two *Text* lacks those glyphs, so
% route each one through math mode where STIX Two Math provides it.
\usepackage{{newunicodechar}}
\newunicodechar{{∈}}{{\ensuremath{{\in}}}}
\newunicodechar{{≈}}{{\ensuremath{{\approx}}}}
\newunicodechar{{∞}}{{\ensuremath{{\infty}}}}
\newunicodechar{{∼}}{{\ensuremath{{\sim}}}}
\newunicodechar{{∣}}{{\ensuremath{{\mid}}}}
\newunicodechar{{←}}{{\ensuremath{{\leftarrow}}}}
\newunicodechar{{→}}{{\ensuremath{{\rightarrow}}}}
\newunicodechar{{∇}}{{\ensuremath{{\nabla}}}}
% Big summation operator defined directly (class 1, math family 0,
% U+2211); any route through \sum loops via the active character.
\Umathchardef\unisum="1 "0 "2211
\newunicodechar{{∑}}{{\ensuremath{{\unisum}}}}
\newunicodechar{{ₜ}}{{\textsubscript{{t}}}}
\newunicodechar{{₀}}{{\ensuremath{{{{}}_{{0}}}}}}
\newunicodechar{{₁}}{{\ensuremath{{{{}}_{{1}}}}}}
\newunicodechar{{₊}}{{\ensuremath{{{{}}_{{+}}}}}}
\newunicodechar{{⁻}}{{\ensuremath{{{{}}^{{-}}}}}}
\newunicodechar{{≥}}{{\ensuremath{{\geq}}}}
\newunicodechar{{𝑇}}{{\ensuremath{{T}}}}
\newunicodechar{{𝜏}}{{\ensuremath{{\tau}}}}
\newunicodechar{{𝜆}}{{\ensuremath{{\lambda}}}}

% --- tables (pandoc uses longtable, booktabs) -----------------------------
\usepackage{{array}}
\usepackage{{longtable}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage[table]{{xcolor}}
\providecommand{{\real}}[1]{{#1}}          % pandoc emits \real{{0.51}}
% Thin grey hairline between body rows so multi-line rows stay readable,
% while the booktabs top/mid/bottom rules remain solid black.
\newcommand{{\rowsep}}{{\arrayrulecolor{{black!22}}\specialrule{{0.35pt}}{{2pt}}{{2.5pt}}\arrayrulecolor{{black}}}}

% --- floats, graphics -----------------------------------------------------
\usepackage{{graphicx}}
\graphicspath{{{{./}}}}
\usepackage{{float}}
\usepackage{{caption}}
\captionsetup{{font=small,labelfont=bf}}

% --- lists, code, hyphenation ---------------------------------------------
\usepackage{{enumitem}}
\usepackage{{ragged2e}}
\usepackage{{parskip}}
\usepackage{{setspace}}

% Boxed algorithm summaries as floats: ruled look, never split, and the
% surrounding text fills the page instead of leaving a gap.
\newenvironment{{algobox}}
  {{\begin{{table}}[!htb]\begin{{minipage}}{{\linewidth}}
   \singlespacing\setlength{{\parskip}}{{2pt}}\small
   \hrule height 0.9pt \vspace{{5pt}}}}
  {{\vspace{{5pt}}\hrule height 0.9pt
   \end{{minipage}}\end{{table}}}}

% --- headers & TOC formatting ---------------------------------------------
\usepackage{{fancyhdr}}
\setlength{{\headheight}}{{14pt}}
\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[L]{{\nouppercase{{\leftmark}}}}
\fancyhead[R]{{\thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
\renewcommand{{\chaptermark}}[1]{{\markboth{{\chaptername\ \thechapter.\ #1}}{{}}}}

\usepackage{{titlesec}}
\titleformat{{\chapter}}[display]
  {{\normalfont\Huge\bfseries}}{{\chaptertitlename\ \thechapter}}{{20pt}}{{\Huge}}
\titlespacing*{{\chapter}}{{0pt}}{{-30pt}}{{25pt}}

% --- hyperlinks -----------------------------------------------------------
\usepackage[hidelinks,pdfusetitle]{{hyperref}}
\hypersetup{{
  colorlinks=false,
  pdftitle={{Evaluating Reinforcement Learning Algorithms for Portfolio Trading}},
  pdfauthor={{Fiyinfoluwa Akano}}
}}

% --- pandoc helper -------------------------------------------------------
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}

% Pandoc emits `\def\LTcaptype{{none}}` around uncaptioned tables. That fake
% counter breaks hyperref, so we shadow it with a real one.
\newcounter{{none}}
\providecommand{{\theHnone}}{{None.\thenone}}

\setcounter{{chapter}}{{2}}   % so the first \chapter{{...}} numbers as Chapter 3

\title{{\Huge Evaluating Reinforcement Learning Algorithms\\
        for Portfolio Trading \\[0.6em]
        \Large A Reinforcement Learning Study on the\\
        S\&P~500 ETF (SPY)}}
\author{{Fiyinfoluwa Akano \\ \small 6962514 \\[0.5em]
        \small MSc Artificial Intelligence \\
        \small School of Computer Science and Electronic Engineering \\
        \small Faculty of Engineering and Physical Sciences \\
        \small University of Surrey}}
\date{{Chapters 3--5 --- \today}}

\begin{{document}}

\maketitle

\begin{{center}}
\vspace*{{1em}}
\textit{{This is a preview build of Chapters 3--5 for supervisor review.\\
Citations, front matter and remaining chapters follow separately.}}
\end{{center}}

\tableofcontents
\clearpage

\chapter{{{ch3_title}}}
\input{{ch3}}

\chapter{{{ch4_title}}}
\input{{ch4}}

\chapter{{{ch5_title}}}
\input{{ch5}}

\renewcommand{{\bibname}}{{References}}
\bibliographystyle{{IEEEtran}}
\bibliography{{references}}

\end{{document}}
"""


if __name__ == "__main__":
    main()
