"""
Streamlit deployment of the Query Router demo. Calls the actual project
modules (categories.py, fireworks_client.py, baseline_router.py,
router/infer_router.py) directly - no separate backend server needed,
Streamlit is both.

Does NOT run the local Qwen model - that was the single biggest disk
(~1GB gguf) and memory consumer in this process, and kept segfaulting
it even after freeing the trained router's memory between queries.
"local"-routed queries are answered by a cheap Fireworks model instead
(MODEL_CHEAP), labeled honestly in the UI as a substitute rather than
passed off as real local inference. The router's own decision is still
real either way - only the answer-generation step changes.

Deploy: push to GitHub, then on share.streamlit.io point the app at
streamlit_app/app.py and add these as Secrets (Settings > Secrets, TOML
format) - same values as your local .env:
    FIREWORKS_API_KEY = "..."
    FIREWORKS_BASE_URL = "..."
    MODEL_EXPENSIVE = "..."
    MODEL_CHEAP = "..."
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Streamlit secrets live in st.secrets, not os.environ - agent.py /
# fireworks_client.py / baseline_router.py all read via os.environ
# directly (same code the Docker image and CI use), so without this
# bridge, secrets set correctly in the app's Settings > Secrets UI are
# still invisible to them.
try:
    for _key in st.secrets:
        os.environ.setdefault(_key, str(st.secrets[_key]))
except Exception:
    pass  # no secrets.toml locally - fine, .env / real env vars cover that case

import categories
import fireworks_client
import baseline_router
from router.infer_router import available as router_available, predict as router_predict

st.set_page_config(page_title="Token-Efficient Query Router", page_icon="🧭", layout="wide")

# Deliberately not `import agent` - agent.py unconditionally imports
# local_llm at its own top level, which needs llama_cpp, which this
# deployment doesn't install (see the module docstring). These two are
# the only things this app actually needs from it, so they're
# duplicated here rather than pulling in that whole import chain just
# to get them.
ROUTER_CONFIDENCE_THRESHOLD = float(os.environ.get("ROUTER_CONFIDENCE_THRESHOLD", "0.65"))


def get_fireworks_model():
    if os.environ.get("MODEL_EXPENSIVE"):
        return os.environ["MODEL_EXPENSIVE"].strip()
    return os.environ["ALLOWED_MODELS"].split(",")[0].strip()

EXAMPLES = [
    "What is 47 + 128?",
    "Convert 5 miles to kilometers",
    "Explain how gradient descent works and why momentum helps convergence",
    "Refactor a nested loop into a comprehension and explain the tradeoffs",
]


def estimate_tokens(s):
    return max(1, round(len(s or "") / 3.6))


def get_cheap_model():
    # This deployment doesn't run the local Qwen model at all - it was the
    # single biggest disk/memory consumer (a ~1GB gguf plus llama-cpp-python's
    # own native footprint) and kept crashing the process even after
    # freeing the router's memory first. "local"-routed queries get
    # answered by a cheap Fireworks model instead, labeled honestly in the
    # UI rather than silently passed off as local.
    val = os.environ.get("MODEL_CHEAP")
    if not val:
        raise KeyError("MODEL_CHEAP")
    return val.strip()


ROUTER_WEIGHTS_URL = "https://github.com/DiphekoK/track1-agent/releases/download/router-weights/model.safetensors"


def ensure_router_weights():
    # config.json/tokenizer files come from git (small), but the actual
    # weights file is a GitHub Release asset (~255MB, too big for a normal
    # git push - same reasoning as the Dockerfile's curl step). A plain
    # git clone, which is what Streamlit Cloud does, never fetches it.
    model_dir = ROOT / "router" / "model"
    weights_path = model_dir / "model.safetensors"
    if weights_path.exists() or not (model_dir / "config.json").exists():
        return
    with st.spinner("Downloading trained router weights (~255MB, first run only)…"):
        try:
            urllib.request.urlretrieve(ROUTER_WEIGHTS_URL, str(weights_path))
        except Exception as e:
            st.info(f"Trained router weights not available ({e}) - using the heuristic instead.")


def get_config():
    try:
        cheap_label = get_cheap_model().rsplit("/", 1)[-1]
        cheap_error = None
    except KeyError as e:
        cheap_label = None
        cheap_error = f"missing env var {e}"
    try:
        fireworks_label = get_fireworks_model().rsplit("/", 1)[-1]
        fireworks_error = None
    except KeyError as e:
        fireworks_label = None
        fireworks_error = f"missing env var {e}"
    return {
        "cheap_model": cheap_label,
        "cheap_error": cheap_error,
        "fireworks_model": fireworks_label,
        "fireworks_error": fireworks_error,
        "router_available": router_available(),
    }


def decide_finetuned(prompt, category):
    confidence, note = None, None
    if router_available():
        label, confidence = router_predict(prompt)
        if confidence < ROUTER_CONFIDENCE_THRESHOLD:
            note = f"router unsure ({confidence:.2f}), used heuristic instead"
            label = "local" if categories.should_use_local(category) else "fireworks"
    else:
        note = "trained router weights not available, used heuristic"
        label = "local" if categories.should_use_local(category) else "fireworks"
    return {"backend": label, "confidence": confidence, "note": note}


def run_baseline(prompt):
    try:
        label, tokens = baseline_router.classify(prompt)
        return {"backend": label, "tokens": tokens, "error": None}
    except Exception as e:
        return {"backend": None, "tokens": 0, "error": str(e)}


def run_answer(prompt, category, backend):
    try:
        system = fireworks_client.SYSTEM_PROMPTS.get(category)
        model = get_cheap_model() if backend == "local" else get_fireworks_model()
        resp = fireworks_client.chat(model, prompt, system=system, max_tokens=400)
        text, tokens = resp["text"], resp["total_tokens"]
        return {"backend": backend, "text": text, "tokens": tokens, "error": None}
    except Exception as e:
        return {"backend": backend, "text": None, "tokens": 0, "error": str(e)}


if "history" not in st.session_state:
    st.session_state.history = []
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

ensure_router_weights()
cfg = get_config()

header_left, header_right = st.columns([2, 1])
with header_left:
    st.caption("AMD DEVELOPER HACKATHON · ACT II · TRACK 1")
    st.title("Token-Efficient Query Router")
    st.write(
        "A fine-tuned local router classifies every query in a zero-token forward pass, "
        "then routes cheap queries to a small model and escalates hard ones. A prompt-based "
        "baseline burns tokens just to make the same decision."
    )
with header_right:
    st.markdown("**MODEL TIERS**")
    st.markdown(f"🟦 **Cheap** &nbsp; `{cfg['cheap_model'] or 'not configured'}` (Fireworks)")
    st.markdown(f"🟧 **Escalation** &nbsp; `{cfg['fireworks_model'] or 'not configured'}` (Fireworks)")

st.info(
    "This hosted demo doesn't run the local Qwen model (too much memory/disk for this tier) - "
    "the router's decision is still real, but 'local'-routed queries are answered by a cheap "
    "Fireworks model instead, labeled as such below rather than passed off as local."
)
if not cfg["router_available"]:
    st.info("Trained router weights not available in this deployment - using the heuristic category rules instead.")
if cfg["cheap_error"]:
    st.warning(f"Cheap-tier model isn't configured ({cfg['cheap_error']}) - local-routed queries will fail until MODEL_CHEAP is set.")
if cfg["fireworks_error"]:
    st.warning(f"Fireworks isn't configured ({cfg['fireworks_error']}) - escalated queries will fail until secrets are set.")

# Placeholders, not direct rendering - render_hero() below gets called
# again after a run appends to history, so the numbers actually reflect
# what just happened instead of staying one run behind (Streamlit
# doesn't retroactively update widgets already sent to the page, so
# computing these once up front and never refreshing them was the bug).
st.divider()
hero_left, hero_right = st.columns([1, 1])
with hero_left:
    hero_caption = st.empty()
    hero_pct = st.empty()
    hero_detail = st.empty()
with hero_right:
    m1, m2, m3, m4 = st.columns(4)
    metric_queries = m1.empty()
    metric_saved = m2.empty()
    metric_ft = m3.empty()
    metric_bl = m4.empty()


def render_hero():
    history = st.session_state.history
    finetuned_total = sum(h["ft"] for h in history)
    baseline_total = sum(h["bl"] for h in history)
    saved = baseline_total - finetuned_total
    saved_pct = round(saved / baseline_total * 100) if baseline_total else 0

    hero_caption.caption("TOKENS SAVED BY FINE-TUNING · THIS SESSION")
    hero_pct.markdown(f"## {saved_pct}% fewer" if baseline_total else "## —")
    hero_detail.caption(
        f"{saved:,} tokens never spent vs. the prompt-based baseline" if baseline_total else "Run a query to see this"
    )
    metric_queries.metric("Queries run", len(history))
    metric_saved.metric("Tokens saved", f"{saved:,}" if baseline_total else "—")
    metric_ft.metric("Fine-tuned total", f"{finetuned_total:,}")
    metric_bl.metric("Baseline total", f"{baseline_total:,}")


render_hero()

st.divider()
st.subheader("Enter a query")
chip_cols = st.columns(len(EXAMPLES))
for col, ex in zip(chip_cols, EXAMPLES):
    label = ex if len(ex) <= 30 else ex[:30] + "…"
    if col.button(label, key=f"ex_{ex}", width="stretch"):
        st.session_state.prompt = ex
        st.rerun()

prompt = st.text_area(
    "Query", value=st.session_state.prompt, height=100, label_visibility="collapsed",
    placeholder="e.g. What is 12 + 7?  ·  or  ·  Explain why transformers scale better than RNNs",
)
run_clicked = st.button("Run through router", type="primary", disabled=not prompt.strip())

st.divider()
st.caption("ROUTING PIPELINE")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**① Fine-tuned router**")
    st.caption("Local forward pass · on-device · 0 tokens")
    status1 = st.empty()
with col2:
    st.markdown("**② Prompt-based baseline**")
    st.caption("Real classifier LLM call · tokens spent to decide")
    status2 = st.empty()
with col3:
    st.markdown("**③ Answer generation**")
    st.caption("Routed to the tier the fine-tuned router chose")
    status3 = st.empty()
status1.caption("Idle")
status2.caption("Idle")
status3.caption("Idle")

if run_clicked:
    q = prompt.strip()
    category = categories.classify(q)

    status1.info("Classifying locally…")
    t0 = time.time()
    ft = decide_finetuned(q, category)
    latency_ms = round((time.time() - t0) * 1000, 1)
    conf_str = f" · {ft['confidence']:.2f} conf" if ft["confidence"] is not None else ""
    status1.success(f"**{ft['backend']}**{conf_str} · {latency_ms}ms · 0 tokens")
    if ft["note"]:
        col1.caption(ft["note"])

    status2.info("Calling classifier…")
    bl = run_baseline(q)
    if bl["error"]:
        status2.error(bl["error"])
    else:
        status2.warning(f"**{bl['backend']}** · {bl['tokens']} tokens")

    status3.info("Generating…")
    ans = run_answer(q, category, ft["backend"])
    if ans["error"]:
        status3.error(ans["error"])
    else:
        model_label = cfg["fireworks_model"] if ans["backend"] == "fireworks" else cfg["cheap_model"]
        substitute_note = " (local unavailable - cheap Fireworks substitute)" if ans["backend"] == "local" else ""
        status3.info(f"**{model_label}**{substitute_note} · ~{ans['tokens']} tokens")

    if ans["text"]:
        st.divider()
        if bl.get("tokens"):
            st.caption(f"💡 {bl['tokens']} tokens the baseline spent just to classify this query — the fine-tuned router decided it for free.")
        st.subheader("Answer")
        st.write(ans["text"])
    elif ans["error"]:
        st.divider()
        st.error(f"Answer generation failed: {ans['error']}")

    answer_tokens = ans["tokens"] or (estimate_tokens(q) + estimate_tokens(ans["text"] or ""))
    st.session_state.history.append({
        "query": q if len(q) <= 60 else q[:60] + "…",
        "backend": ans["backend"] or "?",
        "ft": answer_tokens,
        "bl": (bl.get("tokens") or 0) + answer_tokens,
        "saved": bl.get("tokens") or 0,
    })
    render_hero()  # refresh the placeholders now that this run is in history

history = st.session_state.history
if history:
    st.divider()
    st.subheader("Tokens per query")
    import pandas as pd
    df = pd.DataFrame(history)
    st.bar_chart(df.set_index("query")[["ft", "bl"]].rename(columns={"ft": "Fine-tuned", "bl": "Baseline"}))

    st.subheader("Query log")
    st.dataframe(
        df.rename(columns={"query": "Query", "backend": "Route", "ft": "Fine-tuned", "bl": "Baseline", "saved": "Saved"}),
        width="stretch",
        hide_index=True,
    )
    if st.button("Reset session"):
        st.session_state.history = []
        st.rerun()
