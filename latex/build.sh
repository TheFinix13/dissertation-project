#!/usr/bin/env bash
# Build the dissertation PDF.
#
# Usage:
#   ./build.sh                # one-shot full build (pdflatex, biber, pdflatex, pdflatex)
#   ./build.sh watch          # rebuild on every save (needs latexmk)
#   ./build.sh clean          # remove all generated artifacts
#
# Requirements:
#   - pdflatex + biber on PATH (MacTeX, BasicTeX with `tlmgr install biber`,
#     or Linux texlive-full).
#
set -euo pipefail
cd "$(dirname "$0")"

MAIN=main
ARTIFACTS=(*.aux *.bbl *.bcf *.blg *.log *.out *.toc *.lof *.lot *.run.xml *.fls *.fdb_latexmk)

case "${1:-build}" in
  watch)
    command -v latexmk >/dev/null || { echo "latexmk not installed."; exit 1; }
    latexmk -pdf -pvc -bibtex- -use-biber -interaction=nonstopmode "$MAIN.tex"
    ;;
  clean)
    rm -f "${ARTIFACTS[@]}" "$MAIN.pdf"
    rm -rf chapters/*.aux
    echo "Cleaned."
    ;;
  build|*)
    command -v pdflatex >/dev/null || { echo "pdflatex not installed. See README.md in this folder."; exit 1; }
    command -v biber    >/dev/null || { echo "biber not installed. See README.md in this folder."; exit 1; }
    pdflatex -interaction=nonstopmode "$MAIN.tex"
    biber "$MAIN"
    pdflatex -interaction=nonstopmode "$MAIN.tex"
    pdflatex -interaction=nonstopmode "$MAIN.tex"
    echo ""
    echo "Built $MAIN.pdf ($(wc -c <"$MAIN.pdf" | tr -d ' ') bytes)"
    ;;
esac
