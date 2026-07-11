"""
Runs the local model against every query in queries.jsonl and grades
the answer against its ground truth. Whichever queries the local model
gets wrong become "fireworks" labels for the router to learn from -
whichever it gets right stay "local". This is the measure-don't-guess
step: don't hand-pick which categories are "too hard" for the local
model, actually run it and see.

Each query gets sampled N_SAMPLES times rather than once. local_llm
answers at temperature 0.2, not 0, so a single pass/fail is really a
coin flip for anything borderline - re-running label_dataset.py against
the same queries has previously produced different label counts each
time. Majority-voting across a few samples makes the label closer to
"can the local model reliably handle this" instead of "did it get lucky
this one time", which matters because the router is trained to predict
exactly that label.

No Fireworks account needed for this - all grading here is
programmatic (numeric match, keyword containment, exec test-pass,
exact match), not an LLM judge.

Run after generate_adversarial.py: python data/label_dataset.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import local_llm
from code_exec import run_tests

QUERIES_PATH = Path(__file__).parent / "queries.jsonl"
OUTPUT_PATH = Path(__file__).parent / "labeled_dataset.jsonl"
N_SAMPLES = 5


def grade_numeric(answer_text, ground_truth):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer_text.replace(",", ""))
    target = float(ground_truth)
    return any(abs(float(n) - target) < 1e-6 for n in numbers)


def grade_keywords(answer_text, ground_truth):
    lower = answer_text.lower()
    return all(kw.lower() in lower for kw in ground_truth)


def grade_label(answer_text, ground_truth):
    return ground_truth.lower() in answer_text.lower()


def grade_exact(answer_text, ground_truth):
    return ground_truth.lower() in answer_text.lower()


def grade_exec(answer_text, ground_truth):
    return run_tests(answer_text, ground_truth["function_name"], ground_truth["tests"])


GRADERS = {
    "numeric": grade_numeric,
    "keywords": grade_keywords,
    "label": grade_label,
    "exact": grade_exact,
    "exec": grade_exec,
}


def grade(query, answer_text):
    return GRADERS[query["grading"]](answer_text, query["ground_truth"])


def grade_majority(q, n=N_SAMPLES):
    n_pass = 0
    for _ in range(n):
        try:
            answer_text = local_llm.answer(q["prompt"], q["category"])
            if grade(q, answer_text):
                n_pass += 1
        except Exception as e:
            print(f"[warn] {q['task_id']} errored during grading: {e}", file=sys.stderr)
    return n_pass, n


def main():
    if not QUERIES_PATH.exists():
        print(f"{QUERIES_PATH} doesn't exist - run data/generate_adversarial.py first", file=sys.stderr)
        sys.exit(1)

    queries = [json.loads(line) for line in QUERIES_PATH.read_text().splitlines() if line.strip()]

    labeled = []
    for q in queries:
        n_pass, n = grade_majority(q)
        label = "local" if n_pass > n // 2 else "fireworks"
        labeled.append({"prompt": q["prompt"], "category": q["category"], "label": label})
        print(f"{q['task_id']:>10} [{q['category']:<14}] local {n_pass}/{n} -> {label}")

    OUTPUT_PATH.write_text("\n".join(json.dumps(r) for r in labeled) + "\n")

    n_fireworks = sum(1 for r in labeled if r["label"] == "fireworks")
    print(f"\n{len(labeled)} labeled, {n_fireworks} need fireworks, {len(labeled) - n_fireworks} local is fine")
    print(f"written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
