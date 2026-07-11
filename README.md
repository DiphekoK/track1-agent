# Track 1 agent

Routes each task to either the local model (free) or Fireworks (costs
tokens). The routing decision itself is a small fine-tuned classifier
(`router/`) trained on measured pass/fail data, not a hand-picked list of
"these categories are hard" - see "Why a trained router" below.

## Architecture

- `agent.py` - entry point, reads `input/tasks.json`, writes `output/results.json`
  (override with `INPUT_PATH`/`OUTPUT_PATH` env vars)
- `categories.py` - regex classifier used to pick the right system prompt per
  task, and as a fallback routing rule if the trained router isn't available
- `local_llm.py` - local Qwen2.5-1.5B via llama-cpp-python, the free tier
- `fireworks_client.py` - Fireworks chat wrapper, the paid tier
- `baseline_router.py` - alternative routing: ask a model to classify before
  answering (costs a call per task, kept for comparison)
- `router/infer_router.py` - loads the trained classifier, zero-cost routing
- `router/train_router.py` - fine-tunes DistilBERT on labeled data
- `data/generate_adversarial.py` - builds a labeled query set with verifiable
  ground truth across all 8 categories
- `data/label_dataset.py` - runs the local model against that set, grades it,
  labels each query "local" (local model got it right) or "fireworks"
  (it didn't)

## Why a trained router instead of hand-picked categories

Originally this just hardcoded "math/logic/code go to Fireworks, everything
else stays local" based on a guess about which categories a small model
struggles with. That's exactly the kind of assumption worth checking instead
of shipping: run the local model against a real, verifiable test set and see
what it actually gets wrong, then train a lightweight classifier on that.

Two things fall out of doing it this way:

- If the local model turns out to be fine on some "hard" category, that
  category stops needlessly burning Fireworks tokens.
- If it turns out to be shaky on something assumed "easy", that gets caught
  before the real evaluation does, not after the accuracy gate fails.

## Setup

```
pip install -r requirements.txt
pip install llama-cpp-python==0.3.5 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

Download the local model to the path `local_llm.py` expects:

```
mkdir -p models
curl -L -o models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
export LOCAL_MODEL_PATH="$(pwd)/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"   # Windows: $env:LOCAL_MODEL_PATH
```

## Build the router (run once, or whenever you add more adversarial queries)

```
python data/generate_adversarial.py   # writes data/queries.jsonl, verifies logic puzzles + math while doing it
python data/label_dataset.py          # runs the local model against every query, writes data/labeled_dataset.jsonl
python router/train_router.py         # fine-tunes DistilBERT, saves to router/model/
```

None of these three need a working Fireworks account - grading is all
programmatic (numeric match, keyword containment, exec test-pass, exact
match for logic), not an LLM judge. Watch `label_dataset.py`'s output: if a
whole category comes back 100% "fireworks", the local model is failing it
consistently and might need a better system prompt in `local_llm.py` rather
than just eating the token cost - worth a look before training on it.

`train_router.py` will complain and exit if there's fewer than 10 labeled
records - that just means generate/label didn't run first.

## Run it

Set up real credentials:

```
cp .env.example .env
# fill in FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS, then load them
# into the shell (e.g. a dotenv loader, or export each var manually)
```

Run against the practice tasks in `input/tasks.json`:

```
python agent.py
```

Check `output/results.json` - one `{task_id, answer}` entry per practice
task. Check stderr for `[warn]`/`[error]` lines too.

Sanity checks worth doing before submitting:
- results for practice-02/06/07/08 (math/debug/logic/codegen) look sensible
- if `router/model/` is empty (training pipeline skipped), stderr will show
  `finetuned router weights not found, falling back to heuristic categories`
  - it still runs correctly, just with the old hand-picked category list
  instead of the trained one

## ROUTER_MODE

- `finetuned` (default) - trained classifier, zero-cost routing decision
- `baseline` - asks a Fireworks model to classify first (costs tokens on
  every task just for the routing decision, not just the ones that get
  escalated - kept around to demonstrate why the finetuned router is worth
  having)
- `heuristic` - always use `categories.py`'s hand-picked list, skips the
  trained classifier entirely

## Note on Track 1 submission

The hackathon's own Track 1 requirements specify a Docker image that reads
`/input/tasks.json` and writes `/output/results.json`. This repo currently
has no Dockerfile - if a container submission is needed later, `agent.py`
already reads/writes those paths (via `INPUT_PATH`/`OUTPUT_PATH`, defaulting
to `input/tasks.json`/`output/results.json`), so wrapping it back in a
container is just adding a Dockerfile that copies these files in and sets
those two env vars to `/input/tasks.json` and `/output/results.json`.
