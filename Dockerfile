# Track 1 submission image. Reads /input/tasks.json, writes /output/results.json -
# see the "Note on Track 1 submission" section that used to be in the README.
#
# 3.11 rather than whatever's on the dev machine (3.13) because the
# llama-cpp-python and torch CPU wheels below aren't reliably published for
# every Python version yet, and 3.11 is the safe middle ground for both.
FROM python:3.11-slim-bookworm

# curl to pull the gguf at build time, ca-certificates so that doesn't fail
# on cert verification. Removed again in the same layer so the image doesn't
# carry apt's package lists around for no reason.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# llama-cpp-python and torch aren't in requirements.txt because they each need
# a non-default index (prebuilt CPU wheels - building either from source in a
# container would take forever and needs a compiler toolchain we don't
# otherwise want in the image). Same two commands as the README's local setup.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir llama-cpp-python==0.3.30 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    && pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Baked into the image rather than downloaded at run time - the grading box
# isn't guaranteed to have internet access, and re-downloading a ~1GB file on
# every run would be a waste even if it did. This is also why it's its own
# layer before the COPY below: touching agent.py shouldn't invalidate the
# download and force pulling it again.
RUN mkdir -p models && \
    curl -L -o models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
        https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf

COPY . .

# router/model/'s config and tokenizer files come from the COPY above (small,
# committed to git), but the ~255MB weights file itself is too big for a
# normal git push (GitHub hard-rejects anything over 100MB) so it lives as a
# GitHub Release asset instead, published by train-router.yml. Tolerates a
# 404 here on purpose - if training hasn't been run yet there's no release
# to fetch, and router/infer_router.py's available() check already handles
# a missing weights file by falling back to categories.py's heuristic.
RUN curl -fL -o router/model/model.safetensors \
        https://github.com/DiphekoK/track1-agent/releases/download/router-weights/model.safetensors \
    || echo "no published router weights yet, falling back to the heuristic router"

ENV LOCAL_MODEL_PATH=/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
ENV INPUT_PATH=/input/tasks.json
ENV OUTPUT_PATH=/output/results.json

CMD ["python", "agent.py"]
