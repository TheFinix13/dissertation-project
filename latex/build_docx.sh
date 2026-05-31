#!/usr/bin/env bash
# Build an editable Microsoft Word version of the dissertation.
#
#   ./build_docx.sh            # produces ../dissertation.docx
#
# What it does (the LaTeX sources are NEVER modified):
#   1. Renders the three TikZ diagrams (tikz/*.tex) to standalone PNGs.
#   2. Copies main.tex + chapters into a temporary build dir, swapping each
#      \input{tikz/...} for \includegraphics of the rendered PNG and turning
#      every figure path into an absolute path so pandoc can embed it.
#   3. Runs pandoc (LaTeX -> docx) with --citeproc for in-text citations and a
#      reference list, then a Lua filter relocates that reference list to sit
#      before the appendices with a "References" heading.
#
# Requirements: pandoc, pdflatex, pdftoppm (poppler). The 'standalone' LaTeX
# class is auto-installed via tlmgr if missing (TinyTeX/TeX Live).
set -euo pipefail
cd "$(dirname "$0")"                       # .../latex
LATEX_DIR="$(pwd)"
REPO_DIR="$(cd .. && pwd)"
BUILD="$LATEX_DIR/_docx_build"
FIGS="$BUILD/figs"
OUT="$REPO_DIR/dissertation.docx"

for tool in pandoc pdflatex pdftoppm; do
  command -v "$tool" >/dev/null || { echo "ERROR: '$tool' not found on PATH."; exit 1; }
done

# 'standalone' document class is needed to crop each TikZ picture.
if ! kpsewhich standalone.cls >/dev/null 2>&1; then
  echo "Installing 'standalone' LaTeX class via tlmgr ..."
  tlmgr install standalone >/dev/null 2>&1 || {
    echo "ERROR: could not install 'standalone'. Run: tlmgr install standalone"; exit 1; }
fi

rm -rf "$BUILD"
mkdir -p "$FIGS" "$BUILD/chapters"

# ── 1. Render each TikZ diagram to a cropped PNG ──────────────────────────────
for fig in rl_loop data_splits training_pipeline; do
  {
    printf '%s\n' \
      '\documentclass[border=4pt]{standalone}' \
      '\usepackage{mathpazo}' \
      '\usepackage{amsmath,amssymb}' \
      '\usepackage{tikz}' \
      '\usetikzlibrary{arrows.meta, positioning, calc, shapes.geometric}' \
      '\begin{document}'
    sed -n '/\\begin{tikzpicture}/,/\\end{tikzpicture}/p' "tikz/$fig.tex"
    printf '%s\n' '\end{document}'
  } > "$FIGS/$fig.tex"
  ( cd "$FIGS" && pdflatex -interaction=nonstopmode "$fig.tex" >"$fig.build.log" 2>&1 )
  pdftoppm -png -r 200 "$FIGS/$fig.pdf" "$FIGS/$fig"
  mv "$FIGS/$fig-1.png" "$FIGS/$fig.png"
done

# ── 2. Preprocess the sources into the build dir ──────────────────────────────
LATEX_DIR="$LATEX_DIR" BUILD="$BUILD" FIGS="$FIGS" python3 - <<'PY'
import os, pathlib, re, shutil
LATEX = pathlib.Path(os.environ["LATEX_DIR"])
BUILD = pathlib.Path(os.environ["BUILD"])
FIGS  = pathlib.Path(os.environ["FIGS"])
OUT_CH = BUILD / "chapters"

# tikz name -> (caption, label, width)
TIKZ = {
    "rl_loop": ("One training step of the RL loop: observe, act, receive "
                "reward, update policy.", "fig:rl_loop", "0.7"),
    "data_splits": ("Chronological data splits --- no future information leaks "
                    "into training.", "fig:data_splits", "0.95"),
    "training_pipeline": ("End-to-end training pipeline from raw prices to "
                          "evaluation metrics.", "fig:pipeline", "0.8"),
}

def figure_block(name):
    cap, lab, w = TIKZ[name]
    png = (FIGS / f"{name}.png").resolve()
    return ("\\begin{figure}[htbp]\n\\centering\n"
            f"\\includegraphics[width={w}\\textwidth]{{{png}}}\n"
            f"\\caption{{{cap}}}\n\\label{{{lab}}}\n\\end{{figure}}")

for src in sorted((LATEX / "chapters").glob("*.tex")):
    text = src.read_text(encoding="utf-8")
    for name in TIKZ:
        text = re.sub(r"\\input\{tikz/" + re.escape(name) + r"\}",
                      lambda _m, n=name: figure_block(n), text)
    # absolute-ise chart paths like {../reports/.../x.png}
    text = re.sub(r"\{(\.\./[^}]*\.png)\}",
                  lambda m: "{" + str((LATEX / m.group(1)).resolve()) + "}", text)
    (OUT_CH / src.name).write_text(text, encoding="utf-8")

main = (LATEX / "main.tex").read_text(encoding="utf-8")
main = main.replace(r"\printbibliography[heading=bibintoc]",
                    r"\section*{REFS_PLACEHOLDER}")
(BUILD / "main.tex").write_text(main, encoding="utf-8")
print("preprocessed sources ->", BUILD)
PY

# ── 3. Lua filter: move the reference list before the appendices ──────────────
cat > "$BUILD/move_refs.lua" <<'LUA'
-- Runs AFTER --citeproc. Moves the generated bibliography (Div #refs) from the
-- end of the document to the REFS_PLACEHOLDER marker (before the appendices)
-- and gives it an unnumbered, navigable "References" heading.
function Pandoc(doc)
  local blocks = doc.blocks
  local refs_div, refs_idx
  for i, b in ipairs(blocks) do
    if b.t == "Div" and b.identifier == "refs" then refs_div, refs_idx = b, i end
  end
  if not refs_div then return doc end
  table.remove(blocks, refs_idx)
  local heading = pandoc.Header(1, pandoc.Str("References"),
                                pandoc.Attr("references", {"unnumbered"}))
  for i, b in ipairs(blocks) do
    if b.t == "Header" and
       pandoc.utils.stringify(b):find("REFS_PLACEHOLDER", 1, true) then
      table.remove(blocks, i)
      table.insert(blocks, i, refs_div)
      table.insert(blocks, i, heading)
      return pandoc.Pandoc(blocks, doc.meta)
    end
  end
  table.insert(blocks, heading)
  table.insert(blocks, refs_div)
  return pandoc.Pandoc(blocks, doc.meta)
end
LUA

# ── 4. Convert to docx ────────────────────────────────────────────────────────
( cd "$BUILD" && pandoc main.tex -o "$OUT" \
    --citeproc --bibliography="$LATEX_DIR/references.bib" \
    --lua-filter=move_refs.lua \
    --number-sections --toc )

echo ""
echo "Built $OUT ($(wc -c <"$OUT" | tr -d ' ') bytes)"
