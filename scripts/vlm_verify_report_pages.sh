#!/usr/bin/env bash
# VLM visual verification of every rendered page of the final audit PDF.
set -u
OUT=/home/z/my-project/evidence/vlm_report_pages
mkdir -p "$OUT"

PROMPT='You are auditing one page of a typeset A4 engineering audit report (dark blue cover, light blue body). Judge ONLY rendering quality, not opinions. Check: (1) no text overlaps other text, rules, images or table gridlines; (2) no text is clipped at page edges, margins or cell boundaries; (3) tables are horizontally centered, header row readable (white text on dark blue), no column overflow; (4) charts/figures are intact, uncropped, with readable captions; (5) typography consistent (serif body, no garbled or tofu characters); (6) header/footer present on body pages (title + accent rule at top, author + page number at bottom). Answer in exactly this structure:
LAYOUT: <what this page contains, one line>
OVERLAPS: <any overlapping elements and where; NONE if none>
CLIPPING: <any clipped/cut-off content and where; NONE if none>
TABLES: <table quality if tables present: centered, no overflow; N/A if none>
FIGURES: <figure quality if images present: intact, captioned; N/A if none>
DEFECTS: <any other rendering defect; NONE if none>
VERDICT: <PASS if the page is professionally rendered, FAIL plus reason otherwise>'

total=0
declare -A PAGES
for f in /home/z/my-project/evidence/report_assets/pdf_pages/page_*.png; do
  total=$((total+1)); PAGES[$total]="$f"
done

for idx in $(seq 1 $total); do
  f="${PAGES[$idx]}"
  base=$(basename "$f" .png)
  echo "[$idx/$total] VLM -> $base"
  z-ai vision -p "$PROMPT" -i "$f" -o "$OUT/${base}.json" 2>/dev/null || echo "  !! CLI failed for $base"
done
echo "DONE: $total pages analyzed"
