"""
Calls out to Fireworks for the task categories where the local model
isn't reliable enough (math, logic puzzles, code debug/gen). Uses the
OpenAI SDK since Fireworks exposes an OpenAI-compatible chat endpoint.

Everything here is read from env at call time - no hardcoded keys,
urls or model ids, per the hackathon rules.
"""
import os

from openai import OpenAI

_client = None
_model = None

SYSTEM_PROMPTS = {
    "math": "You are a careful math tutor. Work through the problem step by step, but keep your explanation brief. End with a line that starts with 'Answer:' followed by just the final number or result.",
    "logic": "You solve logic puzzles. Reason through the constraints briefly, then end with a line that starts with 'Answer:' stating the conclusion directly.",
    "code_debug": "You are a senior engineer reviewing code. Identify the bug and provide the corrected implementation. Keep prose short, prioritize the fixed code.",
    "code_gen": "You write correct, well-structured code from a spec. Return the implementation with minimal surrounding explanation.",
}


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ["FIREWORKS_API_KEY"]
        base_url = os.environ["FIREWORKS_BASE_URL"]
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def _get_model():
    global _model
    if _model is None:
        models = os.environ["ALLOWED_MODELS"].split(",")
        _model = models[0].strip()
    return _model


def answer(prompt: str, category: str) -> str:
    client = _get_client()
    model = _get_model()
    system = SYSTEM_PROMPTS.get(category, "Answer the following as accurately and concisely as possible.")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()
