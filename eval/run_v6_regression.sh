#!/bin/zsh
# v6 vs v5 on the frozen 109-question regression suite (same pipeline code as the v5 runs; compare_models.py refuses otherwise) + the 13
# development explanation questions (eval/explain/run4_v6; run1 untouched). Local and free. Runs one model server at a time.
cd ~/newine_local
G=dist/models/newinegpt-4b-v6/newinegpt-4b-v6-Q5_K_M-imat.gguf
[ -f $G ] || { echo "missing $G"; exit 1; }
for q in regression_questions scripture_fu user_questions_regression speaker_questions codex_questions attrib_clarify blind1_regression; do
  [ -f eval/compare/$q/v6.jsonl ] && { echo "skip $q (exists)"; continue; }
  t0=$(date +%s); python3 eval/compare_models.py run v6 $G eval/golden/$q.txt || { echo "FAILED $q"; exit 1; }
  echo "$(date +%T) done $q in $(( $(date +%s) - t0 )) s"
done
EXPLAIN_RUN_DIR=run4_v6 python3 eval/explain_experiment.py run v6 $G && echo "$(date +%T) done explain dev 13"
echo ALL DONE
