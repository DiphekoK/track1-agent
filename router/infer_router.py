"""
Local, zero-token router: decides whether the local model can likely
handle a prompt or whether it should be escalated to Fireworks.

Trained offline (see train_router.py) on real pass/fail data, not
hand-picked category rules. agent.py falls back to categories.py's
heuristic mapping if these weights aren't present - e.g. the first
time the container gets built, before anyone's run the training
pipeline yet.
"""
import os

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

_MODEL_DIR = os.environ.get("ROUTER_MODEL_DIR", os.path.join(os.path.dirname(__file__), "model"))
_LABELS = ["local", "fireworks"]

_tokenizer = None
_model = None


def available():
    return os.path.exists(os.path.join(_MODEL_DIR, "config.json"))


def _load():
    global _tokenizer, _model
    if _model is not None:
        return
    _tokenizer = DistilBertTokenizerFast.from_pretrained(_MODEL_DIR)
    _model = DistilBertForSequenceClassification.from_pretrained(_MODEL_DIR)
    _model.eval()


def predict(prompt):
    _load()
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = _model(**inputs).logits
    idx = int(torch.argmax(logits, dim=1)[0])
    return _LABELS[idx]
