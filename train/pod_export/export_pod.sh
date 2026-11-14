#!/bin/bash
# On the pod after ADAPTER_ONLY training (v6 on; generic form of export.sh used for v5): merge -> bf16 GGUF -> imatrix (calib_v5.txt, the
# same 300 data_v5 training rows as v5, so the quantization recipe is unchanged) -> Q5_K_M-imat, Q4_K_M-imat -> SHA256SUMS.
# No Q8_0 (v5's was never downloaded). Usage: export_pod.sh <name> [base_snapshot_dir_glob]
N=$1; G=/root/out/gguf; mkdir -p $G; exec > /root/export.log 2>&1
fail() { echo "FAILED: $1"; touch /root/export_failed; exit 1; }
until [ -f /root/train_finished ] || [ -f /root/train_failed ]; do sleep 15; done
[ -f /root/train_failed ] && fail training
until [ -f /root/llama_ready ] || [ -f /root/llama_failed ]; do sleep 10; done
[ -f /root/llama_failed ] && fail llama_build
echo "$(date -u +%T) merging"
N=$N /root/v/bin/python - <<'PY' || fail merge
import os, torch, transformers, peft
from huggingface_hub import snapshot_download
base = snapshot_download('Qwen/Qwen3-4B'); ad = f"/root/out/{os.environ['N']}/adapter"
m = transformers.AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map='cuda')
m = peft.PeftModel.from_pretrained(m, ad).merge_and_unload()
m.save_pretrained('/root/merged', safe_serialization=True); transformers.AutoTokenizer.from_pretrained(ad).save_pretrained('/root/merged')
print('merged from', base)
PY
echo "$(date -u +%T) converting"
PYTHONPATH=/root/llama.cpp/gguf-py /root/v/bin/python /root/llama.cpp/convert_hf_to_gguf.py /root/merged --outtype bf16 --outfile /root/$N-bf16.gguf > /root/convert.log 2>&1 || fail convert
B=/root/llama.cpp/build/bin
$B/llama-imatrix -m /root/$N-bf16.gguf -f /root/calib.txt -o $G/imatrix.dat -c 2048 -ngl 99 > /root/imatrix.log 2>&1 || fail imatrix
echo "$(date -u +%T) quantizing"
$B/llama-quantize --imatrix $G/imatrix.dat /root/$N-bf16.gguf $G/$N-Q5_K_M-imat.gguf Q5_K_M > /root/quant.log 2>&1 || fail q5
$B/llama-quantize --imatrix $G/imatrix.dat /root/$N-bf16.gguf $G/$N-Q4_K_M-imat.gguf Q4_K_M >> /root/quant.log 2>&1 || fail q4
(cd $G && sha256sum *.gguf imatrix.dat > SHA256SUMS && cat SHA256SUMS)
echo "$(date -u +%T) export done"; touch /root/export_done
