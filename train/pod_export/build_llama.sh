#!/bin/bash
# Build llama.cpp (A100=80, L40S=89, H100=90; same commit as the Mac: 11fb327bf3846f390e3af1cbe929da9287c618da) with CUDA on the pod, low priority so training is not slowed.
cd /root && git init -q llama.cpp && cd llama.cpp && git remote add origin https://github.com/ggml-org/llama.cpp && git fetch -q --depth 1 origin 11fb327bf3846f390e3af1cbe929da9287c618da && git checkout -q FETCH_HEAD || { touch /root/llama_failed; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH
nice -n 19 cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;89;90" -DLLAMA_CURL=OFF > /root/cmake.log 2>&1 && nice -n 19 cmake --build build --target llama-quantize llama-imatrix -j $(( $(nproc) < 16 ? $(nproc) : 16 )) >> /root/cmake.log 2>&1 && touch /root/llama_ready || touch /root/llama_failed
