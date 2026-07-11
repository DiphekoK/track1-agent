"""
Entry point for the Track 1 agent.

Reads /input/tasks.json, routes each task to either the local model
(free) or Fireworks (costs tokens, only used when we think it's worth
it), and writes /output/results.json before exiting.

Kept as one script rather than a bigger framework - this is a batch
job that runs once and exits, doesn't need much more structure than
this.
"""
import json
import os
import sys
import time
import traceback

import router
import local_llm
import fireworks_client

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"

# leave a few minutes of buffer under the 10 minute hard limit so we
# always have time to write out whatever we've got instead of getting
# killed mid-task
MAX_RUNTIME_SECONDS = 8.5 * 60


def load_tasks():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_results(results):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    tmp_path = OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, OUTPUT_PATH)


def process_task(task, start_time):
    task_id = task["task_id"]
    prompt = task["prompt"]
    category = router.classify(prompt)
    use_local = router.should_use_local(category)

    try:
        if use_local:
            ans = local_llm.answer(prompt, category)
        else:
            ans = fireworks_client.answer(prompt, category)
        return {"task_id": task_id, "answer": ans}
    except Exception:
        print(f"[warn] primary backend failed for {task_id} (category={category}):", file=sys.stderr)
        traceback.print_exc()

    # primary backend blew up - try the other one rather than giving up
    try:
        if use_local:
            ans = fireworks_client.answer(prompt, category)
        else:
            ans = local_llm.answer(prompt, category)
        return {"task_id": task_id, "answer": ans}
    except Exception:
        print(f"[warn] fallback backend also failed for {task_id}:", file=sys.stderr)
        traceback.print_exc()
        return {"task_id": task_id, "answer": ""}


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
        results.append(process_task(task, start_time))

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
