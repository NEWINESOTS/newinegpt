# newineGPT public demo (Hugging Face Space, free CPU). Uses the published GitHub release: the Linux app (bundled Python, llama.cpp,
# archive, search model) and the v6 Q4 model, each checked against the release's SHA256SUMS.txt; the three app files copied here add the
# public mode (no key, conversations kept in each visitor's browser, one answer at a time).
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl unzip && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user
USER user
WORKDIR /home/user
ARG REL=https://github.com/NEWINESOTS/newinegpt/releases/download/v0.6.0-beta
RUN curl -fsSL -O $REL/SHA256SUMS.txt \
 && for f in newineGPT-beta-linux-x64.zip newinegpt-4b-v6-Q4_K_M-imat-00001-of-00002.gguf newinegpt-4b-v6-Q4_K_M-imat-00002-of-00002.gguf; do curl -fsSL -O $REL/$f; done \
 && grep -E 'linux-x64.zip|Q4_K_M' SHA256SUMS.txt | sha256sum -c - \
 && unzip -q newineGPT-beta-linux-x64.zip && rm newineGPT-beta-linux-x64.zip \
 && mv newinegpt-4b-v6-Q4_K_M-imat-0000*.gguf newineGPT/dist/models/newinegpt-4b-v6/
COPY --chown=user ask.py web.py web_index.html newineGPT/app/newinegpt/
WORKDIR /home/user/newineGPT
ENV NEWINEGPT_PUBLIC=1 PORT=7860 NEWINEGPT_THREADS=2 PYTHONDONTWRITEBYTECODE=1
EXPOSE 7860
CMD ["runtime/linux-x64/python/bin/python3", "-X", "utf8", "start.py", "--q4", "--no-browser"]
