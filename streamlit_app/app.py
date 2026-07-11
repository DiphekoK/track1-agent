"""
Streamlit deployment of the Query Router demo. Unlike the earlier
Cloudflare/static-HTML attempt, Streamlit Community Cloud runs a real
Python process, so this calls the actual project modules
(categories.py, local_llm.py, fireworks_client.py, baseline_router.py,
router/infer_router.py) directly - no separate backend server needed,
Streamlit is both.

Deploy: push to GitHub, then on share.streamlit.io point the app at
streamlit_app/app.py and add these as Secrets (Settings > Secrets, TOML
format) - same values as your local .env:
    FIREWORKS_API_KEY = "..."
    FIREWORKS_BASE_URL = "..."
    MODEL_EXPENSIVE = "..."

The local model (~1GB gguf) isn't in git - this downloads it on first
run if it's not already present. Free-tier resource limits are a real
risk for a 1.5B model plus the Streamlit process; if it doesn't fit,
local-tier queries will fail over to Fireworks automatically (same
fallback-on-error agent.py already has), not crash the app.
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

MODEL_URL = "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
os.environ.setdefault("LOCAL_MODEL_PATH", str(ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"))

import categories
import local_llm
import fireworks_client
import baseline_router
import agent
from router.infer_router import available as router_available, predict as router_predict

st.set_page_config(page_title="Token-Efficient Query Router", page_icon="🧭", layout="wide")

EXAMPLES = [
    "What is 47 + 128?",
    "Convert 5 miles to kilometers",
    "Explain how gradient descent works and why momentum helps convergence",
    "Refactor a nested loop into a comprehension and explain the tradeoffs",
]


def estimate_tokens(s):
    return max(1, round(len(s or "") / 3.6))


def ensure_local_model():
    path = Path(local_llm.MODEL_PATH)
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner(f"Downloading the local model (~1GB, first run only)…"):
        try:
            urllib.request.urlretrieve(MODEL_URL, str(path))
            return True
        except Exception as e:
            st.warning(f"Couldn't download the local model ({e}) - local-tier queries will fail over to Fireworks.")
            return False


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
    local_label = Path(local_llm.MODEL_PATH).stem
    try:
        fireworks_label = agent.get_fireworks_model().rsplit("/", 1)[-1]
        fireworks_error = None
    except KeyError as e:
        fireworks_label = None
        fireworks_error = f"missing env var {e}"
    return {
        "local_model": local_label,
        "fireworks_model": fireworks_label,
        "fireworks_error": fireworks_error,
        "router_available": router_available(),
    }


def decide_finetuned(prompt, category):
    confidence, note = None, None
    if router_available():
        label, confidence = router_predict(prompt)
        if confidence < agent.ROUTER_CONFIDENCE_THRESHOLD:
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
        if backend == "local":
            text, tokens = local_llm.answer_with_usage(prompt, category)
            if tokens is None:
                tokens = estimate_tokens(prompt) + estimate_tokens(text)
        else:
            system = fireworks_client.SYSTEM_PROMPTS.get(category)
            resp = fireworks_client.chat(agent.get_fireworks_model(), prompt, system=system, max_tokens=400)
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
    st.markdown(f"🟦 **Local** &nbsp; `{cfg['local_model']}`")
    st.markdown(f"🟧 **Fireworks** &nbsp; `{cfg['fireworks_model'] or 'not configured'}`")

if not cfg["router_available"]:
    st.info("Trained router weights not available in this deployment - using the heuristic category rules instead.")
if cfg["fireworks_error"]:
    st.warning(f"Fireworks isn't configured ({cfg['fireworks_error']}) - escalated queries will fail until secrets are set.")

history = st.session_state.history
finetuned_total = sum(h["ft"] for h in history)
baseline_total = sum(h["bl"] for h in history)
saved = baseline_total - finetuned_total
saved_pct = round(saved / baseline_total * 100) if baseline_total else 0

st.divider()
hero_left, hero_right = st.columns([1, 1])
with hero_left:
    st.caption("TOKENS SAVED BY FINE-TUNING · THIS SESSION")
    st.markdown(f"## {saved_pct}% fewer" if baseline_total else "## —")
    st.caption(f"{saved:,} tokens never spent vs. the prompt-based baseline" if baseline_total else "Run a query to see this")
with hero_right:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Queries run", len(history))
    m2.metric("Tokens saved", f"{saved:,}" if baseline_total else "—")
    m3.metric("Fine-tuned total", f"{finetuned_total:,}")
    m4.metric("Baseline total", f"{baseline_total:,}")

st.divider()
st.subheader("Enter a query")
chip_cols = st.columns(len(EXAMPLES))
for col, ex in zip(chip_cols, EXAMPLES):
    label = ex if len(ex) <= 30 else ex[:30] + "…"
    if col.button(label, key=f"ex_{ex}", use_container_width=True):
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

    if categories.should_use_local(category):
        ensure_local_model()

    status1.info("Classifying locally…")
    t0 = time.time()
    ft = decide_finetuned(q, category)
    latency_ms = round((time.time() - t0) * 1000, 1)
    conf_str = f" · {ft['confidence']:.2f} conf" if ft["confidence"] is not None else ""
    status1.success(f"**{ft['backend']}**{conf_str} · {latency_ms}ms · 0 tokens")
    if ft["note"]:
        col1.caption(ft["note"])

    if ft["backend"] == "local":
        ensure_local_model()

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
        model_label = cfg["fireworks_model"] if ans["backend"] == "fireworks" else cfg["local_model"]
        status3.info(f"**{model_label}** · ~{ans['tokens']} tokens")

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

if history:
    st.divider()
    st.subheader("Tokens per query")
    import pandas as pd
    df = pd.DataFrame(history)
    st.bar_chart(df.set_index("query")[["ft", "bl"]].rename(columns={"ft": "Fine-tuned", "bl": "Baseline"}))

    st.subheader("Query log")
    st.dataframe(
        df.rename(columns={"query": "Query", "backend": "Route", "ft": "Fine-tuned", "bl": "Baseline", "saved": "Saved"}),
        use_container_width=True,
        hide_index=True,
    )
    if st.button("Reset session"):
        st.session_state.history = []
        st.rerun()
