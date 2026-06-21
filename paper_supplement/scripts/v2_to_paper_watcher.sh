#!/bin/bash
# v2 sweep completion watcher — polls v2_sweep TSV, when all 180 units DONE
# triggers analyze_v2.py + appends Section 7 v2 draft into hand-off brief.

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TSV="$REPO/SimMTM_Classification/outputs/v2_sweep/run_status.tsv"
LOG="$REPO/paper_supplement/scripts/v2_watcher.log"
TRIG="$REPO/paper_supplement/scripts/v2_triggered.flag"

echo "watcher start $(date -Iseconds)" > "$LOG"
EXPECTED=180  # 2 backbones × 6 dt × 3 seeds × (pretrain + c4 + 3×c5) = 180

while true; do
  # Count terminal states (DONE + FAIL) — sweep is "done" when all 180 attempted.
  done=$(tail -n +2 "$TSV" 2>/dev/null | awk -F'\t' '$2=="DONE" || $2 ~ /^FAIL/' | wc -l || echo 0)
  done_ok=$(tail -n +2 "$TSV" 2>/dev/null | awk -F'\t' '$2=="DONE"' | wc -l || echo 0)
  echo "[$(date -Iseconds)] v2_sweep terminal: $done/$EXPECTED (DONE=$done_ok)" >> "$LOG"
  if [ "$done" -ge "$EXPECTED" ]; then
    echo "[$(date -Iseconds)] v2 sweep complete — triggering analyze_v2" >> "$LOG"
    /usr/bin/python3 "$REPO/paper_supplement/scripts/analyze_v2.py" >> "$LOG" 2>&1
    date -Iseconds > "$TRIG"
    echo "[$(date -Iseconds)] analyze_v2 complete" >> "$LOG"
    break
  fi
  sleep 900  # poll every 15 minutes
done
