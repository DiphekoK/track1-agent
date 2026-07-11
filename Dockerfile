FROM python:3.11-slim

WORKDIR /app

# prebuilt CPU wheel for llama-cpp-python - avoids pulling in a full
# c++ toolchain just to compile it at build time
RUN pip install --no-cache-dir llama-cpp-python==0.3.5 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# local model, ~1GB, quantized so it fits comfortably in the 4GB
# grading box alongside the rest of the process
RUN mkdir -p /app/models
ADD https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf /app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf

COPY app/ /app/

ENTRYPOINT ["python", "main.py"]
