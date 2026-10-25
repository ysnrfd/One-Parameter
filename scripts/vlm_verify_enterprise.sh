#!/usr/bin/env bash
# VLM verification of the ENTERPRISE-build GUI frames (post-fix re-audit).
set -u
OUT=/home/z/my-project/evidence/vlm_enterprise
mkdir -p "$OUT"

PROMPT='You are auditing one rendered frame of a scientific matplotlib dashboard called the "single float32 parameter observatory". The figure should contain four panels: top-left a vertical "causal chain" diagram (INPUT -> TRACKED PARAMETER w -> ... -> NEW PARAMETER, with one box highlighted orange for the active phase UPDATE), top-right a "recorded series (live)" line plot with four series (w, delta w, gradient, loss), a purple circular marker and trail, a small FPS readout in the top-right corner of that panel, and an interpolation notice at its bottom-left; bottom-left a 32-bit float32 bit-pattern strip (grey/orange squares labelled 0/1, titled "stored float32 bits after step N"); bottom-right a monospace "state card" HUD listing step, w, hex, delta, attempted, loss, gradient, lr, recon, bits changed, finite. Audit this frame forensically and answer in exactly this structure:
LAYOUT: <which of the four panels are present and correctly placed>
SERIES: <what the line plot shows: how many series, do data points/lines exist, is the purple marker visible, approximate trend>
TEXT: <key text values you can read: step number, w value, loss, FPS readout, bit count>
BITS: <how many bit squares are drawn, which are orange (changed), the hex word if readable>
DEFECTS: <any rendering defects: overlapping/clipped text, empty panels, missing series, garbled characters, misaligned elements; write NONE if none>
VERDICT: <PASS if the frame is a coherent, correctly rendered dashboard frame; FAIL plus reason otherwise>'

total=0
declare -A FRAMES
for f in /home/z/my-project/evidence/frames_gui_enterprise/frame_*.png; do
  total=$((total+1)); FRAMES[$total]="$f"
done
for f in /home/z/my-project/single_parameter_observatory/run_checks_frames_ink/frame_*.png; do
  total=$((total+1)); FRAMES[$total]="$f"
done

for idx in $(seq 1 $total); do
  f="${FRAMES[$idx]}"
  base=$(basename "$f" .png)
  if [[ "$f" == *frames_gui_enterprise* ]]; then tag="ent_${base}"; else tag="ink_${base}"; fi
  echo "[$idx/$total] VLM -> $tag"
  z-ai vision -p "$PROMPT" -i "$f" -o "$OUT/${tag}.json" 2>/dev/null || echo "  !! CLI failed for $tag"
done
echo "DONE: $total frames analyzed"
