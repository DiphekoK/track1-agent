"""
Calls out to Fireworks for whatever the router decided the local model
can't handle. Uses the OpenAI SDK since Fireworks exposes an
OpenAI-compatible chat endpoint.

Also used by data/label_dataset.py and baseline_router.py, which is why
chat() hands back token usage instead of just the answer text - the
labeling script needs it to compare cost across approaches.
"""
import os

from openai import OpenAI

_client = None

SYSTEM_PROMPTS = {
    "math": "You are a careful math tutor. Work through the problem step by step, but keep your explanation brief. End with a line that starts with 'Answer:' followed by just the final number or result.",
    "logic": "You solve logic puzzles. Reason through the constraints briefly, then end with a line that starts with 'Answer:' stating the conclusion directly.",
    "code_debug": "You are a senior engineer reviewing code. Identify the bug and provide the corrected implementation. Keep prose short, prioritize the fixed code.",
    "code_gen": "You write correct, well-structured code from a spec. Return the implementation with minimal surrounding explanation.",
    "factual": "You are a helpful, accurate assistant. Answer clearly and concisely. If the question has multiple parts, answer all of them.",
    "sentiment": "You classify sentiment. Reply with the label (positive, negative, or neutral) followed by a one-sentence justification.",
    "summarization": "You summarize text exactly as instructed (length/format constraints in the prompt must be followed precisely).",
    "ner": "You extract named entities from text. List each entity with its type (PERSON, ORG, LOCATION, DATE, etc).",
}


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ["FIREWORKS_BASE_URL"]
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def chat(model, prompt, system=None, max_tokens=400, temperature=0.2):
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {
        "text": resp.choices[0].message.content.strip(),
        "total_tokens": resp.usage.total_tokens,
    }
