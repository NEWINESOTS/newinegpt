#!/bin/zsh
# Deploy one GPU, push the training job, arm a hard cap (pod-side killswitch + Mac-side terminate), sync /root/out back, terminate when finished/failed.
# Usage: train/launch_train.sh <cap_hours> <HF base id> <name> "<quants>"   e.g. train/launch_train.sh 3 Qwen/Qwen3-4B newinegpt-4b-v1 Q4_K_M
CAP_H=${1:-3}; CAP_S=$(printf %.0f $((CAP_H*3600))); BASE=${2:-Qwen/Qwen3-4B}; NAME=${3:-newinegpt-4b-v1}; QUANTS=${4:-Q4_K_M}
# Dataset must be chosen explicitly (GPT audit 12 Nov: the launcher silently used train/data = v3). DATA=data_v5 train/launch_train.sh ...
DATA=${DATA:?set DATA=<folder under train/>, e.g. DATA=data_v5}
for f in train valid test; do [ -s "$(dirname $0)/$DATA/$f.jsonl" ] || { echo "missing $DATA/$f.jsonl"; exit 1; }; done
DATA_DESC=$(cd "$(dirname $0)/$DATA" && for f in train valid test; do printf "%s=%s " $f $(wc -l < $f.jsonl | tr -d ' '); done)
echo "dataset: $DATA ($DATA_DESC)"
T=~/newine_local/train; E=$T; LOG=$T/launch_$NAME.log; mkdir -p $T/pod_out/$NAME
set -a; . ${RUNPOD_ENV:-$HOME/.config/newinegpt/runpod.env}; set +a; PUB=$(cat ~/.ssh/id_ed25519.pub)
gql() { curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'content-type: application/json' https://api.runpod.io/graphql -d "$1"; }
terminate() { gql "{\"query\":\"mutation{ podTerminate(input:{podId:\\\"$1\\\"}) }\"}" >/dev/null; echo "$(date +%T) terminated $1" >> $LOG; }
GPU_LIST=("NVIDIA L40S" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" "NVIDIA RTX 6000 Ada Generation" "NVIDIA L40" "NVIDIA H100 PCIe" "NVIDIA RTX PRO 6000 Blackwell Server Edition")
[ -n "$GPUS" ] && GPU_LIST=("${(@s:,:)GPUS}")   # override: GPUS="NVIDIA A100 80GB PCIe,NVIDIA H100 PCIe"
POD=""
for attempt in $(seq 1 20); do
  for GPU in "${GPU_LIST[@]}"; do
    BODY=$(python3 -I -c 'import json,sys
q="mutation($i: PodFindAndDeployOnDemandInput){ podFindAndDeployOnDemand(input:$i){ id costPerHr } }"
i={"cloudType":"SECURE","gpuCount":1,"gpuTypeId":sys.argv[1],"name":"newine-train","imageName":"runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404","containerDiskInGb":100,"volumeInGb":0,"minVcpuCount":4,"minMemoryInGb":32,"ports":"22/tcp","startSsh":True,"env":[{"key":"PUBLIC_KEY","value":sys.argv[2]}]}
print(json.dumps({"query":q,"variables":{"i":i}}))' "$GPU" "$PUB")
    RES=$(gql "$BODY"); POD=$(echo "$RES" | python3 -I -c 'import json,sys
try: print(json.load(sys.stdin)["data"]["podFindAndDeployOnDemand"]["id"])
except Exception: pass')
    [ -n "$POD" ] && { echo "$(date +%T) deployed $POD $GPU $(echo $RES | grep -o '"costPerHr":[0-9.]*')" >> $LOG; break 2; }
  done
  echo "$(date +%T) attempt $attempt: no GPU, waiting 60s" >> $LOG; sleep 60
done
[ -z "$POD" ] && { echo "$(date +%T) GAVE UP" >> $LOG; exit 1; }
(sleep $((CAP_S+300)); terminate $POD) >/dev/null 2>&1 &   # NB: ending this sleep early (pkill) terminates the pod at once
for i in $(seq 1 90); do
  J=$(gql "{\"query\":\"query{ pod(input:{podId:\\\"$POD\\\"}){ runtime{ ports{ ip publicPort privatePort } } } }\"}")
  read HOST PORT <<< $(echo "$J" | python3 -I -c 'import json,sys
try:
    p=[x for x in json.load(sys.stdin)["data"]["pod"]["runtime"]["ports"] if x["privatePort"]==22][0]; print(p["ip"], p["publicPort"])
except Exception: pass')
  [ -n "$HOST" ] && break; sleep 5; done
[ -z "$HOST" ] && { echo "$(date +%T) no SSH port" >> $LOG; terminate $POD; exit 1; }
echo "POD=$POD\nHOST=$HOST\nPORT=$PORT" > $T/pod_$NAME.env
O=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ConnectionAttempts=3 -i ~/.ssh/id_ed25519)
SSH=(ssh -n $O -p $PORT root@$HOST)
for i in $(seq 1 60); do $SSH true 2>/dev/null && break; sleep 5; done
$SSH "mkdir -p /root/data"
for f in train.jsonl valid.jsonl test.jsonl; do for t in 1 2 3 4 5; do scp -q $O -P $PORT $T/$DATA/$f root@$HOST:/root/data/ && break; sleep 5; done; done
# verify the pod has exactly the dataset we chose (sizes + sha256), else stop before any training time is spent
for f in train.jsonl valid.jsonl test.jsonl; do
  L=$(shasum -a 256 $T/$DATA/$f | cut -d' ' -f1); R=$($SSH "sha256sum /root/data/$f" | cut -d' ' -f1)
  [ "$L" = "$R" ] || { echo "$(date +%T) DATA MISMATCH $f local=$L pod=$R" >> $LOG; terminate $POD; exit 1; }
done
echo "$(date +%T) data verified on pod: $DATA ($DATA_DESC)" >> $LOG
for f in pod_train.py pod_train.sh prompt.py; do for t in 1 2 3 4 5; do scp -q $O -P $PORT $T/$f root@$HOST:/root/ && break; sleep 5; done; done
# template is in the project (v5 run: the NAS copy was unmounted and an empty killswitch went to the pod)
sed "s/__SECS__/$CAP_S/" $T/killswitch.tmpl > $E/killswitch.sh; [ -s $E/killswitch.sh ] || { echo "$(date +%T) empty killswitch; terminating" >> $LOG; terminate $POD; exit 1; }
scp -q $O -P $PORT $E/killswitch.sh root@$HOST:/root/killswitch.sh
cat > $T/_start_runner.sh <<EOS
#!/bin/bash
chmod +x /root/killswitch.sh /root/pod_train.sh
ps -eo args | grep -v grep | grep -q "^/bin/bash /root/killswitch.sh" || (nohup setsid /root/killswitch.sh >/dev/null 2>&1 </dev/null &) >/dev/null 2>&1
export BASE=$BASE NAME=$NAME QUANTS="$QUANTS" EPOCHS=${EPOCHS:-2} ADAPTER_ONLY=${ADAPTER_ONLY:-0} MAXLEN=${MAXLEN:-2048} BASE_SNAPSHOT=${BASE_SNAPSHOT:-}
ps -eo args | grep -v grep | grep -q "^/bin/bash /root/pod_train.sh" || (nohup setsid /root/pod_train.sh >/root/train.log 2>&1 </dev/null &) >/dev/null 2>&1
sleep 3; echo "running: \$(ps -eo args | grep -v grep | grep -cE '^/bin/bash /root/(killswitch|pod_train).sh')/2"
EOS
OK=""
for t in $(seq 1 10); do
  scp -q $O -P $PORT $T/_start_runner.sh root@$HOST:/root/start_runner.sh 2>/dev/null && R=$(timeout 60 ssh -n $O -p $PORT root@$HOST 'bash /root/start_runner.sh' 2>/dev/null)
  echo "$(date +%T) start attempt $t: $R" >> $LOG; [ "$R" = "running: 2/2" ] && { OK=1; break; }; sleep 10
done
[ -z "$OK" ] && { echo "$(date +%T) could not start runner; terminating" >> $LOG; terminate $POD; exit 1; }
echo "$(date +%T) started on $POD $HOST:$PORT (cap ${CAP_H}h)" >> $LOG
# HANDOFF=1: a separate finish script (pod_export/finish.sh) waits for the pod-side export and downloads; the cap subshell above stays armed
[ -n "$HANDOFF" ] && { echo "$(date +%T) finish loop handed off" >> $LOG; exit 0; }
RS=(rsync -az --partial --exclude ckpt -e "ssh $O -p $PORT")
while true; do
  sleep 180
  $RS --exclude '*.gguf' root@$HOST:/root/out/ $T/pod_out/$NAME/ 2>/dev/null
  ST=$(ssh -n $O -p $PORT root@$HOST 'tail -c 400 /root/train.log | tr "\r\n" "  "; ls /root/train_finished /root/train_failed 2>/dev/null' 2>/dev/null)
  echo "$(date +%T) $ST" >> $LOG
  if echo "$ST" | grep -q train_failed; then scp -q $O -P $PORT "root@$HOST:/root/*.log" $T/pod_out/$NAME/ 2>/dev/null; terminate $POD; echo FAILED >> $LOG; exit 1; fi
  if echo "$ST" | grep -q train_finished; then
    for t in 1 2 3 4 5; do $RS root@$HOST:/root/out/ $T/pod_out/$NAME/ && break; sleep 10; done
    scp -q $O -P $PORT "root@$HOST:/root/*.log" $T/pod_out/$NAME/ 2>/dev/null
    (cd $T/pod_out/$NAME && shasum -a 256 -c SHA256SUMS >> $LOG 2>&1) && { terminate $POD; echo "$(date +%T) FINISHED" >> $LOG; exit 0; }
    echo "$(date +%T) checksum mismatch after download; retrying" >> $LOG
  fi
done
