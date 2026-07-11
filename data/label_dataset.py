"""
Runs the local model against every query in queries.jsonl and grades
the answer against its ground truth. Whichever queries the local model
gets wrong become "fireworks" labels for the router to learn from -
whichever it gets right stay "local". This is the measure-don't-guess
step: don't hand-pick which categories are "too hard" for the local
model, actually run it and see.

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


def main():
    if not QUERIES_PATH.exists():
        print(f"{QUERIES_PATH} doesn't exist - run data/generate_adversarial.py first", file=sys.stderr)
        sys.exit(1)

    queries = [json.loads(line) for line in QUERIES_PATH.read_text().splitlines() if line.strip()]

    labeled = []
    for q in queries:
        try:
            answer_text = local_llm.answer(q["prompt"], q["category"])
            passed = grade(q, answer_text)
        except Exception as e:
            print(f"[warn] {q['task_id']} errored during grading: {e}", file=sys.stderr)
            passed = False
        label = "local" if passed else "fireworks"
        labeled.append({"prompt": q["prompt"], "category": q["category"], "label": label})
        print(f"{q['task_id']:>10} [{q['category']:<14}] local {'PASS' if passed else 'FAIL'} -> {label}")

    OUTPUT_PATH.write_text("\n".join(json.dumps(r) for r in labeled) + "\n")

    n_fireworks = sum(1 for r in labeled if r["label"] == "fireworks")
    print(f"\n{len(labeled)} labeled, {n_fireworks} need fireworks, {len(labeled) - n_fireworks} local is fine")
    print(f"written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
