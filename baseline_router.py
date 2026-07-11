"""
Prompt-based fallback for deciding local vs fireworks: ask a model to
classify the query before answering it. Works, but every request now
costs an extra classification call - the whole point of
router/infer_router.py's fine-tuned classifier is to make that
decision free instead. Kept around for comparison and as a fallback
when ROUTER_MODE=baseline.
"""
import os

import fireworks_client

CLASSIFY_PROMPT = """Classify the following query as either "local" or \
"fireworks". Answer "local" if a small, general-purpose language model \
could answer it correctly on its own. Answer "fireworks" if it needs a \
larger, more capable model (multi-step math, logic puzzles, non-trivial \
code). Respond with exactly one word: local or fireworks.

Query: {prompt}"""


def _classify_model():
    if os.environ.get("MODEL_CLASSIFY"):
        return os.environ["MODEL_CLASSIFY"]
    return os.environ["ALLOWED_MODELS"].split(",")[0].strip()


def classify(prompt):
    # max_tokens needs real headroom - reasoning models spend tokens
    # thinking before they output the actual word, cut this too low
    # (e.g. 10) and it never gets there, silently defaulting to "local"
    # every time.
    result = fireworks_client.chat(_classify_model(), CLASSIFY_PROMPT.format(prompt=prompt), max_tokens=150)
    label = "fireworks" if "fireworks" in result["text"].lower() else "local"
    return label, result["total_tokens"]
