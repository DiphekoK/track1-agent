"""
Local, zero-token router: decides whether the local model can likely
handle a prompt or whether it should be escalated to Fireworks.

Trained offline (see train_router.py) on real pass/fail data, not
hand-picked category rules. agent.py falls back to categories.py's
heuristic mapping if these weights aren't present - e.g. the first
time the container gets built, before anyone's run the training
pipeline yet.
"""
import gc
import os

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

_MODEL_DIR = os.environ.get("ROUTER_MODEL_DIR", os.path.join(os.path.dirname(__file__), "model"))
_LABELS = ["local", "fireworks"]

_tokenizer = None
_model = None


def available():
    # config.json alone isn't enough - it's committed to git, but the
    # weights file is a separate ~255MB download from a GitHub Release
    # (too big for a normal git push) that might not have landed yet.
    return (
        os.path.exists(os.path.join(_MODEL_DIR, "config.json"))
        and os.path.exists(os.path.join(_MODEL_DIR, "model.safetensors"))
    )


def _load():
    global _tokenizer, _model
    if _model is not None:
        return
    _tokenizer = DistilBertTokenizerFast.from_pretrained(_MODEL_DIR)
    _model = DistilBertForSequenceClassification.from_pretrained(_MODEL_DIR)
    _model.eval()


def predict(prompt):
    """Returns (label, confidence). Confidence is the softmax probability
    of the predicted class - agent.py uses it to decide whether to trust
    this or fall back to the heuristic, since a classifier trained on a
    few hundred examples is going to have plenty of near-50/50 calls that
    are safer left to the hand-picked category rules."""
    _load()
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs))
    return _LABELS[idx], float(probs[idx])


def unload():
    """Frees the loaded model/tokenizer so they're not resident at the
    same time as a memory-heavy neighbor (the Streamlit demo's local
    llama.cpp model, both in the same process). Costs a few seconds to
    reload on the next predict() call. Doesn't reclaim torch/transformers'
    own base import overhead, just the actual weights/tensors - if a
    process is still tight after this, that overhead is the remainder."""
    global _tokenizer, _model
    _tokenizer = None
    _model = None
    gc.collect()
