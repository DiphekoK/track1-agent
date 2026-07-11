"""
Entry point for the Track 1 agent.

Reads input/tasks.json, routes each task to either the local model
(free) or Fireworks (costs tokens), and writes output/results.json
before exiting. Paths are overridable via INPUT_PATH/OUTPUT_PATH env
vars if this ever needs to run somewhere with a different layout.

Routing decision comes from ROUTER_MODE:
  finetuned (default) - router/infer_router.py's trained classifier,
                         a local forward pass, zero tokens spent deciding
  baseline            - ask a Fireworks model to classify first, costs
                         an extra call per task (see baseline_router.py)
  heuristic           - categories.py's hand-picked category list, used
                         automatically if finetuned weights aren't there
"""
import json
import os
import sys
import time
import traceback

import categories
import local_llm
import fireworks_client
import baseline_router

INPUT_PATH = os.environ.get("INPUT_PATH", "input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "output/results.json")

# leave a few minutes of buffer under the 10 minute hard limit so we
# always have time to write out whatever we've got instead of getting
# killed mid-task
MAX_RUNTIME_SECONDS = 8.5 * 60

ROUTER_MODE = os.environ.get("ROUTER_MODE", "finetuned")


def get_fireworks_model():
    # MODEL_EXPENSIVE lets you point at a specific model during your own
    # testing; on submission day there's no such env var, so this falls
    # back to whatever the harness put in ALLOWED_MODELS
    if os.environ.get("MODEL_EXPENSIVE"):
        return os.environ["MODEL_EXPENSIVE"].strip()
    return os.environ["ALLOWED_MODELS"].split(",")[0].strip()


def decide_backend(prompt, category):
    if ROUTER_MODE == "baseline":
        label, _ = baseline_router.classify(prompt)
        return label

    if ROUTER_MODE == "finetuned":
        from router.infer_router import available, predict
        if available():
            return predict(prompt)
        print("[warn] finetuned router weights not found, falling back to heuristic categories", file=sys.stderr)

    return "local" if categories.should_use_local(category) else "fireworks"


def answer_with(backend, prompt, category):
    if backend == "local":
        return local_llm.answer(prompt, category)
    system = fireworks_client.SYSTEM_PROMPTS.get(category)
    return fireworks_client.chat(get_fireworks_model(), prompt, system=system, max_tokens=400)["text"]


def process_task(task):
    task_id = task["task_id"]
    prompt = task["prompt"]
    category = categories.classify(prompt)
    backend = decide_backend(prompt, category)

    try:
        return {"task_id": task_id, "answer": answer_with(backend, prompt, category)}
    except Exception:
        print(f"[warn] {backend} backend failed for {task_id} (category={category}):", file=sys.stderr)
        traceback.print_exc()

    # primary backend blew up - try the other one rather than giving up
    other = "fireworks" if backend == "local" else "local"
    try:
        return {"task_id": task_id, "answer": answer_with(other, prompt, category)}
    except Exception:
        print(f"[warn] fallback backend also failed for {task_id}:", file=sys.stderr)
        traceback.print_exc()
        return {"task_id": task_id, "answer": ""}


def load_tasks():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(results):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    tmp_path = OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, OUTPUT_PATH)


def main():
    start_time = time.time()

    try:
        tasks = load_tasks()
    except Exception:
        print(f"[error] couldn't read {INPUT_PATH}:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    results = []
    for task in tasks:
        if time.time() - start_time > MAX_RUNTIME_SECONDS:
            print(f"[warn] approaching runtime limit, stopping early at {len(results)}/{len(tasks)} tasks", file=sys.stderr)
            break
        results.append(process_task(task))

    try:
        write_results(results)
    except Exception:
        print(f"[error] couldn't write {OUTPUT_PATH}:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    print(f"done: {len(results)}/{len(tasks)} tasks written to {OUTPUT_PATH}")
    sys.exit(0)


if __name__ == "__main__":
    main()
