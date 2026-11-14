#!/bin/zsh
# Generic finish loop for pod-side export runs (v6 on; replaces finish_v5.sh): poll every 60 s; when /root/export_done appears, download
# adapter + logs + imatrix, then the Q5_K_M and Q4_K_M imat GGUFs as parallel byte ranges (v5's single rsync stream took 29 min of paid time),
# verify every file's sha256 against the pod's SHA256SUMS, then terminate. A file that fails verification is repaired with rsync (only
# differing blocks are re-sent). Hard cap: terminate at DEADLINE (epoch secs) no matter what.  Usage: finish.sh <name> <deadline_epoch>
setopt pipefail
N=$1; DEADLINE=$2; PARTS=${PARTS:-4}; T=~/newine_local/train; LOG=$T/launch_$N.log; D=$T/pod_out/$N; mkdir -p $D/gguf
set -a; . ${RUNPOD_ENV:-$HOME/.config/newinegpt/runpod.env}; . $T/pod_$N.env; set +a
gql() { curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' https://api.runpod.io/graphql -d "$1"; }
terminate() { gql "{\"query\":\"mutation{ podTerminate(input:{podId:\\\"$POD\\\"}) }\"}" >/dev/null; echo "$(date +%T) terminated $POD ($1)" >> $LOG; }
(sleep $((DEADLINE - $(date +%s))); terminate "hard cap") >/dev/null 2>&1 &
O=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -i ~/.ssh/id_ed25519)
RS=(rsync -a --partial --exclude ckpt -e "ssh $O -p $PORT")
part() {   # bytes [i*per MiB, (i+1)*per MiB) of the pod's gguf/$1 into the same offset of the local file; retried whole
  local f=$1 i=$2 per=$3 t
  for t in 1 2 3 4 5; do
    ssh -n $O -p $PORT root@$HOST "dd if=/root/out/gguf/$f bs=1M skip=$((i*per)) count=$per status=none" \
      | dd of=$D/gguf/$f bs=1M seek=$((i*per)) conv=notrunc status=none && return 0
    echo "$(date +%T) part $f/$i failed (try $t)" >> $LOG; sleep 5
  done; return 1
}
fetch() {  # one GGUF in $PARTS parallel ranges
  local f=$1 size per i
  size=$(ssh -n $O -p $PORT root@$HOST "stat -c %s /root/out/gguf/$f") || return 1
  per=$(( (size / 1048576 + PARTS) / PARTS )); : > $D/gguf/$f
  for i in $(seq 0 $((PARTS - 1))); do part $f $i $per & done; wait
  [ "$(stat -f %z $D/gguf/$f)" = "$size" ] || { echo "$(date +%T) $f size $(stat -f %z $D/gguf/$f) != $size" >> $LOG; return 1; }
}
verify() { (cd $D && shasum -a 256 -c SHA256SUMS && cd gguf && shasum -a 256 -c SHA256SUMS) >> $LOG 2>&1; }
while true; do
  sleep 60
  ST=$(ssh -n $O -p $PORT root@$HOST 'tail -c 160 /root/train.log | tr "\r\n" "  "; tail -1 /root/export.log 2>/dev/null; ls /root/export_done /root/export_failed /root/train_failed /root/llama_failed 2>/dev/null' 2>/dev/null)
  echo "$(date +%T) $ST" >> $LOG
  if echo "$ST" | grep -qE 'export_failed|train_failed'; then
    $RS --exclude '*.gguf' root@$HOST:/root/out/ $D/ 2>/dev/null; scp -q $O -P $PORT "root@$HOST:/root/*.log" $D/ 2>/dev/null
    terminate failed; echo "$(date +%T) FAILED" >> $LOG; exit 1; fi
  if echo "$ST" | grep -q export_done; then
    t0=$(date +%s)
    for t in 1 2 3; do $RS --exclude '*.gguf' root@$HOST:/root/out/ $D/ && break; sleep 5; done
    scp -q $O -P $PORT "root@$HOST:/root/*.log" $D/ 2>/dev/null
    echo "$(date +%T) downloading GGUFs ($PARTS ranges each, in parallel)" >> $LOG
    # wait for the two fetches by pid: a bare `wait` also waited for the hard-cap timer subshell above (v6 run, 13 Nov: the download
    # finished but the script would have sat until the cap, paid). Fixed after that run.
    fetch $N-Q5_K_M-imat.gguf & p1=$!; fetch $N-Q4_K_M-imat.gguf & p2=$!; wait $p1 $p2
    echo "$(date +%T) downloaded in $(( $(date +%s) - t0 )) s" >> $LOG
    for t in 1 2 3; do
      verify && { terminate finished; echo "$(date +%T) FINISHED (download $(( $(date +%s) - t0 )) s)" >> $LOG; exit 0; }
      echo "$(date +%T) checksum mismatch; repairing with rsync (try $t)" >> $LOG
      $RS --no-whole-file root@$HOST:/root/out/ $D/ 2>>$LOG
    done
    echo "$(date +%T) still failing after repairs; leaving pod up for inspection (cap at $(date -r $DEADLINE +%T))" >> $LOG; exit 2
  fi
done
