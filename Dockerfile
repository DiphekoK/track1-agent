FROM python:3.11-slim

WORKDIR /app

# prebuilt CPU wheel for llama-cpp-python - avoids pulling in a full
# c++ toolchain just to compile it at build time
RUN pip install --no-cache-dir llama-cpp-python==0.3.5 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# CPU-only torch build - the default PyPI wheel drags in CUDA libs we
# don't need and don't want counted against the 10GB image cap
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# local model, ~1GB, quantized so it fits comfortably in the 4GB
# grading box alongside the rest of the process
RUN mkdir -p /app/models
ADD https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf /app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

COPY agent.py categories.py local_llm.py fireworks_client.py baseline_router.py ./
COPY router/infer_router.py ./router/infer_router.py
COPY router/model/ ./router/model/

ENTRYPOINT ["python", "agent.py"]
