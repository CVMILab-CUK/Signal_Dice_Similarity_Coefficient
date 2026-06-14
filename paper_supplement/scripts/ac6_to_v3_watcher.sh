#!/bin/bash
# Watcher: waits for AC-6 60/60 DONE, then triggers V3 model selection.
TSV=SimMTM_Forecasting/outputs/experiments/ac6_anchor/run_status.tsv
LOG=paper_supplement/scripts/ac6_v3_watcher.log
TRIG=paper_supplement/scripts/v3_triggered.flag
echo "watcher start $(date -Iseconds)" > "$LOG"
while true; do
  done=$(tail -n +2 "$TSV" 2>/dev/null | awk -F'\t' '$5=="DONE"' | wc -l)
  echo "[$(date -Iseconds)] AC-6 DONE: $done/60" >> "$LOG"
  if [ "$done" -ge 60 ]; then
    echo "[$(date -Iseconds)] AC-6 complete — triggering V3 + AC-6 analyzer" >> "$LOG"
    # Run both analyzers
    /usr/bin/python3 paper_supplement/scripts/v3_model_selection.py >> "$LOG" 2>&1
    /usr/bin/python3 paper_supplement/scripts/analyze_ac6_anchor.py >> "$LOG" 2>&1
    date -Iseconds > "$TRIG"
    echo "[$(date -Iseconds)] V3 + AC-6 analyses complete" >> "$LOG"
    break
  fi
  sleep 600   # poll every 10 min
done
