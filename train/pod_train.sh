#!/bin/bash
# Runs on the pod: install -> download base -> train LoRA -> merge -> test generations -> GGUF (bf16 -> Q4_K_M[,Q5_K_M]) -> checksums.
# Env: BASE (HF id), NAME (output name), QUANTS ("Q4_K_M" or "Q4_K_M Q5_K_M"), EPOCHS
cd /root
export PATH=/root/v/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1 PIP_ROOT_USER_ACTION=ignore
fail() { echo "FAILED: $1"; touch /root/train_failed; exit 1; }
(/usr/bin/python3 -m pip install -q --break-system-packages --ignore-installed uv && /usr/bin/python3 -m uv venv --clear -p /usr/bin/python3 --system-site-packages /root/v && \
 /usr/bin/python3 -m uv pip install --python /root/v/bin/python --no-deps "transformers>=4.51,<4.58" "peft>=0.13,<0.18" "accelerate>=1.0,<1.11" "gguf>=0.10" && \
 /usr/bin/python3 -m uv pip install --python /root/v/bin/python "tokenizers>=0.21,<0.23" "safetensors>=0.4.3" "huggingface_hub>=0.34,<1.0" hf_transfer regex pyyaml tqdm \
   packaging psutil filelock requests sentencepiece protobuf) > /root/pip.log 2>&1 || fail pip   # torch/torchvision: the image's own CUDA build, never a second copy
/root/v/bin/python -c "import torch,torchvision,transformers,peft,accelerate; assert torch.__file__.startswith('/usr/'), torch.__file__; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'transformers',transformers.__version__,'peft',peft.__version__)" || fail torch
echo "$(date -u +%T) installed"
/root/v/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$BASE'))" > /root/dl.log 2>&1 || fail download
echo "$(date -u +%T) base downloaded"
# optional pin: stop before any training time if the hub's base revision differs from the one earlier versions used
[ -n "$BASE_SNAPSHOT" ] && { grep -q "snapshots/$BASE_SNAPSHOT" /root/dl.log || fail "base_revision (want $BASE_SNAPSHOT, got $(tail -1 /root/dl.log))"; }
if [ "$ADAPTER_ONLY" = "1" ]; then   # cheapest path: train, save adapter, checksum, done (merge/GGUF/tests on the Mac)
  mkdir -p /root/out
  /root/v/bin/python pod_train.py --base "$BASE" --data /root/data --out /root/out/$NAME --merged /root/merged --epochs ${EPOCHS:-2} --maxlen ${MAXLEN:-2048} --adapter_only || fail train
  (cd /root/out && sha256sum $NAME/adapter/adapter_model.safetensors > SHA256SUMS && cat SHA256SUMS)
  echo "$(date -u +%T) adapter done"; touch /root/train_finished; exit 0
fi
# build llama.cpp tools in the background while training runs
(git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp && cd /root/llama.cpp && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF >/root/cmake.log 2>&1 \
  && cmake --build build --target llama-quantize -j 4 >>/root/cmake.log 2>&1 && touch /root/quantize_ready) &
mkdir -p /root/out
/root/v/bin/python pod_train.py --base "$BASE" --data /root/data --out /root/out/$NAME --merged /root/merged --epochs ${EPOCHS:-2} --maxlen ${MAXLEN:-2048} || fail train
echo "$(date -u +%T) trained+merged"
wait
[ -f /root/quantize_ready ] || fail llama_build
/root/v/bin/python /root/llama.cpp/convert_hf_to_gguf.py /root/merged --outtype bf16 --outfile /root/$NAME-bf16.gguf > /root/convert.log 2>&1 || fail convert
for Q in $QUANTS; do /root/llama.cpp/build/bin/llama-quantize /root/$NAME-bf16.gguf /root/out/$NAME-$Q.gguf $Q >> /root/quant.log 2>&1 || fail quant_$Q; done
(cd /root/out && sha256sum *.gguf > SHA256SUMS && cat SHA256SUMS)
echo "$(date -u +%T) gguf done"
touch /root/train_finished
