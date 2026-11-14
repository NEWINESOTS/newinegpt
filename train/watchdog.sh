#!/bin/zsh
# Stall watchdog (alert only, never terminates): every 2 min, prints ALERT if no file under /root changed in STALL_MIN minutes,
# or the runner is not running, or SSH fails. The hard cap (killswitch) still bounds cost. Ends when the pod is gone.
NAME=${1:-newinegpt-4b-v1}; STALL_MIN=${2:-12}; T=~/newine_local/train; LOG=$T/launch_$NAME.log
set -a; . ${RUNPOD_ENV:-$HOME/.config/newinegpt/runpod.env}; . $T/pod_$NAME.env; set +a
gql() { curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' https://api.runpod.io/graphql -d "$1"; }
O=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -i ~/.ssh/id_ed25519); miss=0
while true; do
  sleep 120
  alive=$(gql "{\"query\":\"query{ pod(input:{podId:\\\"$POD\\\"}){ id desiredStatus } }\"}" | grep -o '"desiredStatus":"[A-Z]*"')
  [ -z "$alive" ] && { echo "$(date +%T) watchdog: pod gone, exiting" >> $LOG; exit 0; }
  AGE=$(ssh -n $O -p $PORT root@$HOST 'now=$(date +%s); new=$(find /root/*.log /root/out /root/*.gguf /root/merged -type f -printf "%T@\n" 2>/dev/null | sort -n | tail -1 | cut -d. -f1); echo $((now-new)); nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; ls /root/train_finished /root/train_failed >/dev/null 2>&1 && echo 1 || (ps -eo args | grep -v grep | grep -c "/root/pod_train.sh")' 2>/dev/null | tr '\n' ' ')
  if [ -z "$AGE" ]; then miss=$((miss+1)); echo "$(date +%T) watchdog: ssh failed ($miss)" >> $LOG
    [ $miss -ge 3 ] && echo "ALERT $(date +%T) pod unreachable for $((miss*2)) min"
    continue; fi
  miss=0; set -- ${=AGE}; echo "$(date +%T) watchdog: last file change ${1}s ago, gpu ${2}%" >> $LOG; echo "ok $(date +%T) last change ${1}s ago, gpu ${2}%"
  [ "$1" -gt $((STALL_MIN*60)) ] && echo "ALERT $(date +%T) no progress for ${1}s (gpu ${2}%)"
  [ "$3" = "0" ] && echo "ALERT $(date +%T) runner not running"
done
