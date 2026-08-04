#!/usr/bin/env bash
# Build NEW Chapter 4 / Chapter 5 Word docs from the simple-modelling LaTeX.
# Does NOT touch the University Word skeletons outside this repo.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
CHARTS="$ROOT/reports/generated/charts"
TIKZ="$ROOT/latex/tikz"
TMP="$(mktemp -d)"

fix_png() {
  # Append .png to \includegraphics{name} that lack an extension
  sed -E 's/\\includegraphics(\[[^]]*\])\{([^.{}]+)\}/\\includegraphics\1\{\2.png}/g' "$1"
}

fix_png "$DIR/ch4_implementation_simple.tex" > "$TMP/ch4.tex"
fix_png "$DIR/ch5_results_simple.tex" > "$TMP/ch5.tex"

pandoc "$TMP/ch4.tex" -f latex -t docx \
  --resource-path="$DIR:$CHARTS:$TIKZ" \
  -o "$DIR/Chapter_4_Implementation.docx"

pandoc "$TMP/ch5.tex" -f latex -t docx \
  --resource-path="$DIR:$CHARTS:$TIKZ" \
  -o "$DIR/Chapter_5_Empirical_Results.docx"

# Combined (kept for convenience)
cat "$TMP/ch4.tex" "$TMP/ch5.tex" > "$TMP/ch45.tex"
pandoc "$TMP/ch45.tex" -f latex -t docx \
  --resource-path="$DIR:$CHARTS:$TIKZ" \
  -o "$DIR/Chapter4-5_simple_modelling.docx"

ls -lh "$DIR"/Chapter*.docx
rm -rf "$TMP"
echo "Done. Original University skeletons were not modified."
