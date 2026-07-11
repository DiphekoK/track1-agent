"""
Thin wrapper around llama-cpp-python so the rest of the code doesn't
need to know about model paths, ctx sizes etc.

Model is loaded once and reused for every task (loading it per-task
would be way too slow). Kept small (1.5B, Q4_K_M) on purpose so it
stays inside the 4GB RAM grading box with room left for the container
itself and the Python process.
"""
import os

from llama_cpp import Llama

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")

_llm = None


def _thread_count():
    override = os.environ.get("LOCAL_LLM_THREADS")
    if override:
        return int(override)
    # os.cpu_count() can misreport on constrained/virtualized runners (CI
    # containers especially) - an oversubscribed llama.cpp tends to crawl
    # rather than error out, which just looks like a hang from the
    # outside. Capping it is cheap insurance; CI runners are 2-4 cores
    # anyway so this isn't leaving real performance on the table there.
    return min(os.cpu_count() or 2, 4)

SYSTEM_PROMPTS = {
    "factual": "You are a helpful, accurate assistant. Answer clearly and concisely. If the question has multiple parts, answer all of them.",
    "sentiment": "You classify sentiment. Reply with the label (positive, negative, or neutral) followed by a one-sentence justification.",
    "summarization": "You summarize text exactly as instructed (length/format constraints in the prompt must be followed precisely).",
    "ner": "You extract named entities from text. List each entity with its type (PERSON, ORG, LOCATION, DATE, etc).",
}


def _get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH,
            # the KV cache scales directly with n_ctx - at 2048 that's a
            # few hundred MB on top of the ~1GB model weights, more
            # headroom than the practice tasks or demo prompts need.
            # Configurable rather than just lowered, since the real
            # grading harness's prompts are an unknown that shouldn't be
            # gambled on - only the memory-constrained Streamlit demo
            # overrides this down.
            n_ctx=int(os.environ.get("LOCAL_LLM_N_CTX", "2048")),
            n_threads=_thread_count(),
            verbose=False,
        )
    return _llm


def _generate(prompt: str, category: str):
    llm = _get_llm()
    system = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["factual"])

    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    text = out["choices"][0]["message"]["content"].strip()
    # llama-cpp-python mirrors the OpenAI response shape, usage included -
    # real token count instead of estimating, same as fireworks_client.chat
    total_tokens = out.get("usage", {}).get("total_tokens")
    return text, total_tokens


def answer(prompt: str, category: str) -> str:
    text, _ = _generate(prompt, category)
    return text


def answer_with_usage(prompt: str, category: str):
    """Same as answer(), but also returns the real token count (used by
    web/server.py's demo, which needs it for the tokens-per-query chart)."""
    return _generate(prompt, category)
