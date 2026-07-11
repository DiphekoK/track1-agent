"""
Builds data/queries.jsonl: a set of prompts across all 8 hackathon
categories, each with a ground truth that's checkable by code rather
than eyeballed - numeric answers computed here in Python, logic
puzzles brute-forced for a unique solution before being written down,
code tasks graded by running real test cases (see code_exec.py).

factual/summarization/ner don't have a clean "one correct string"
ground truth, so those use a required-keywords check instead of an
LLM judge - looser than a real judge, but it means the whole labeling
pipeline runs without needing a working Fireworks account at all.

Run directly: python data/generate_adversarial.py
"""
import itertools
import json
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "queries.jsonl"

QUERIES = []
_next_id = {"n": 0}


def _q(category, prompt, ground_truth, grading):
    _next_id["n"] += 1
    QUERIES.append({
        "task_id": f"adv-{_next_id['n']:03d}",
        "category": category,
        "prompt": prompt,
        "ground_truth": ground_truth,
        "grading": grading,
    })


def _solve_unique(names, items, constraints):
    """Brute-forces every assignment of items to names and returns the
    one that satisfies every constraint. Blows up loudly if the puzzle
    as written doesn't have exactly one solution - better to find that
    out here than after it's already in the training set."""
    solutions = []
    for perm in itertools.permutations(items):
        assignment = dict(zip(names, perm))
        if all(c(assignment) for c in constraints):
            solutions.append(assignment)
    assert len(solutions) == 1, f"puzzle not uniquely solvable ({len(solutions)} solutions): {names}/{items}"
    return solutions[0]


# ---------------------------------------------------------------- math
# ground truths computed here, not hand-typed, so arithmetic mistakes
# fail loudly instead of silently mislabeling the dataset

_q("math",
   "A tank starts with 480 liters. It drains at 8 liters per minute for 15 "
   "minutes, then is refilled at 12 liters per minute for 20 minutes, then "
   "drains again at 5 liters per minute for 10 minutes. How many liters "
   "are in the tank now?",
   str(480 - 8 * 15 + 12 * 20 - 5 * 10), "numeric")

_q("math",
   "A store has 350 items. It sells 20% on Monday and 45 more on Tuesday. "
   "How many items remain?",
   str(350 - round(350 * 0.20) - 45), "numeric")

_q("math",
   "A pool starts empty. It's filled at 25 liters per minute for 12 "
   "minutes, then drained at 10 liters per minute for 5 minutes, then "
   "filled at 15 liters per minute for 8 minutes. How many liters are in "
   "the pool now?",
   str(25 * 12 - 10 * 5 + 15 * 8), "numeric")

_q("math",
   "A seller buys widgets for $8 each and sells them for $15 each. After "
   "selling 42 widgets and paying $60 in fixed expenses, what is the "
   "total profit?",
   str((15 - 8) * 42 - 60), "numeric")

_q("math",
   "What is the average of these numbers: 12, 18, 9, 25, 31, 7?",
   str(sum([12, 18, 9, 25, 31, 7]) / 6), "numeric")

_q("math",
   "A price starts at $200. It increases by 25%, then decreases by 10%. "
   "What is the final price?",
   str(200 * 1.25 * 0.9), "numeric")

_q("math",
   "One worker assembles 5 units per hour and works 6 hours. A second "
   "worker assembles 3 units per hour and works 9 hours. How many units "
   "did they assemble in total?",
   str(5 * 6 + 3 * 9), "numeric")

_q("math",
   "An item costs $150. It's discounted 10%, then an additional 5% off "
   "the already-discounted price. What is the final price?",
   str(150 * 0.9 * 0.95), "numeric")

# ------------------------------------------------------------- sentiment

for template, product, label in [
    ("The {p} exceeded expectations, works flawlessly every time.", "laptop stand", "positive"),
    ("The {p} broke after two days and support never replied.", "blender", "negative"),
    ("The {p} does exactly what the description says, nothing more, nothing less.", "wireless mouse", "neutral"),
    ("Best {p} I've bought this year, worth every cent.", "office chair", "positive"),
    ("Complete waste of money, the {p} stopped working within a week.", "coffee maker", "negative"),
    ("The {p} arrived on time and matches the listing.", "running shoes", "neutral"),
    ("I'm impressed by how well the {p} performs given the price.", "desk lamp", "positive"),
    ("Regret buying this {p}, it's noisy and poorly built.", "backpack", "negative"),
]:
    _q("sentiment", f"Classify the sentiment of this review: {template.format(p=product)}", label, "label")

# ------------------------------------------------------------------ ner

for prompt, entities in [
    ("Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March.",
     ["Maria Sanchez", "Fireworks AI", "Berlin", "March"]),
    ("Extract all named entities and their types from: David Kim signed a partnership with Nordic Robotics in Oslo on July 14, 2025.",
     ["David Kim", "Nordic Robotics", "Oslo", "July 14"]),
    ("Extract all named entities and their types from: Amara Okafor relocated to Nairobi to lead engineering at Vantage Health in January.",
     ["Amara Okafor", "Nairobi", "Vantage Health", "January"]),
    ("Extract all named entities and their types from: Liam O'Connor presented the roadmap for Solstice Analytics in Dublin this September.",
     ["Liam O'Connor", "Solstice Analytics", "Dublin", "September"]),
    ("Extract all named entities and their types from: Priya Nair started as CTO of Meridian Labs in Singapore in 2024.",
     ["Priya Nair", "Meridian Labs", "Singapore", "2024"]),
    ("Extract all named entities and their types from: Carlos Mendes opened a new office for Bright Path Robotics in Lisbon last November.",
     ["Carlos Mendes", "Bright Path Robotics", "Lisbon", "November"]),
]:
    _q("ner", prompt, entities, "keywords")

# ---------------------------------------------------------- summarization

for text, keywords in [
    ("The city council voted on Tuesday to approve a new bike lane network "
     "spanning twelve kilometres, with construction expected to begin in "
     "spring and finish before next year's transit summit.",
     ["bike lane", "twelve", "spring"]),
    ("Researchers at the university published a study showing that the "
     "local bee population declined by eighteen percent over the past "
     "five years, largely due to habitat loss and pesticide use.",
     ["bee", "eighteen", "habitat"]),
    ("The airline announced it will add three new direct routes to South "
     "America starting next quarter, citing rising demand from business "
     "travellers.",
     ["airline", "South America", "routes"]),
    ("A local bakery chain is expanding into two neighbouring towns after "
     "securing a small business loan, with the first new location "
     "expected to open by the end of the year.",
     ["bakery", "loan", "expand"]),
]:
    _q("summarization", f"Summarize the following in exactly one sentence: {text}", keywords, "keywords")

# --------------------------------------------------------------- factual

for prompt, keywords in [
    ("What is the capital of Australia, and what body of water is it near?",
     ["Canberra", "Lake Burley Griffin"]),
    ("What is photosynthesis?", ["light", "glucose", "oxygen"]),
    ("What causes rainbows to form?", ["light", "water"]),
    ("What is the difference between TCP and UDP?", ["connection", "UDP"]),
    ("Who developed the theory of general relativity, and in what year was it published?",
     ["Einstein", "1915"]),
    ("What is the boiling point of water at sea level in Celsius?", ["100"]),
]:
    _q("factual", prompt, keywords, "keywords")

# ----------------------------------------------------------------- logic

_a = _solve_unique(
    ["Sam", "Jo", "Lee"], ["cat", "dog", "bird"],
    [lambda a: a["Sam"] != "bird", lambda a: a["Jo"] == "dog"],
)
_q("logic",
   "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, "
   "bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?",
   [n for n, v in _a.items() if v == "cat"][0], "exact")

_b = _solve_unique(
    ["Ivy", "Noah", "Mia"], ["red", "blue", "green"],
    [lambda a: a["Ivy"] != "red", lambda a: a["Noah"] == "green"],
)
_q("logic",
   "Three friends, Ivy, Noah, and Mia, each drive a different colored "
   "car: red, blue, green. Ivy's car isn't red. Noah's car is green. Who "
   "drives the blue car?",
   [n for n, v in _b.items() if v == "blue"][0], "exact")

_c = _solve_unique(
    ["Kai", "Lena", "Theo", "Ren"], ["tea", "coffee", "juice", "water"],
    [lambda a: a["Kai"] != "tea", lambda a: a["Kai"] != "water",
     lambda a: a["Lena"] == "coffee", lambda a: a["Theo"] != "water"],
)
_q("logic",
   "Four friends, Kai, Lena, Theo, and Ren, each drink a different "
   "beverage: tea, coffee, juice, water. Kai doesn't drink tea or water. "
   "Lena drinks coffee. Theo doesn't drink water. Who drinks water?",
   [n for n, v in _c.items() if v == "water"][0], "exact")

_e = _solve_unique(
    ["Nora", "Pablo", "Yuki"], ["piano", "violin", "drums"],
    [lambda a: a["Nora"] != "drums", lambda a: a["Pablo"] == "violin"],
)
_q("logic",
   "Three musicians, Nora, Pablo, and Yuki, each play a different "
   "instrument: piano, violin, drums. Nora doesn't play drums. Pablo "
   "plays violin. Who plays the drums?",
   [n for n, v in _e.items() if v == "drums"][0], "exact")

_f = _solve_unique(
    ["Grace", "Hugo", "Elin"], ["mango", "kiwi", "fig"],
    [lambda a: a["Grace"] != "fig", lambda a: a["Hugo"] == "kiwi"],
)
_q("logic",
   "Three friends, Grace, Hugo, and Elin, each pick a different fruit: "
   "mango, kiwi, fig. Grace didn't pick the fig. Hugo picked kiwi. Who "
   "picked the fig?",
   [n for n, v in _f.items() if v == "fig"][0], "exact")

_g = _solve_unique(
    ["Owen", "Priya", "Sana", "Tariq"], ["tennis", "chess", "rugby", "golf"],
    [lambda a: a["Owen"] != "chess", lambda a: a["Owen"] != "golf",
     lambda a: a["Priya"] == "rugby", lambda a: a["Sana"] != "golf"],
)
_q("logic",
   "Four friends, Owen, Priya, Sana, and Tariq, each play a different "
   "sport: tennis, chess, rugby, golf. Owen doesn't play chess or golf. "
   "Priya plays rugby. Sana doesn't play golf. Who plays golf?",
   [n for n, v in _g.items() if v == "golf"][0], "exact")

# ----------------------------------------------------------- code_debug

_q("code_debug",
   "This function should return the max of a list but has a bug:\n"
   "def get_max(nums): return nums[0]\n"
   "Find and fix it.",
   {"function_name": "get_max", "tests": [
       {"args": [[3, 1, 4, 1, 5, 9, 2, 6]], "expected": 9},
       {"args": [[-5, -1, -3]], "expected": -1},
       {"args": [[7]], "expected": 7},
   ]}, "exec")

_q("code_debug",
   "This function should sum all numbers from 1 to n inclusive but has a "
   "bug:\ndef sum_to_n(n):\n    total = 0\n    for i in range(1, n):\n"
   "        total += i\n    return total\nFind and fix it.",
   {"function_name": "sum_to_n", "tests": [
       {"args": [5], "expected": 15},
       {"args": [1], "expected": 1},
       {"args": [10], "expected": 55},
   ]}, "exec")

_q("code_debug",
   "This function should append to a fresh list each call but has a "
   "bug:\ndef add_item(item, items=[]):\n    items.append(item)\n"
   "    return items\nFind and fix it so repeated calls don't share "
   "state.",
   {"function_name": "add_item", "tests": [
       {"args": ["a"], "expected": ["a"]},
       {"args": ["b"], "expected": ["b"]},
   ]}, "exec")

_q("code_debug",
   "This function should check if a number is even but has a bug:\n"
   "def is_even(n):\n    return n % 2 == 1\nFind and fix it.",
   {"function_name": "is_even", "tests": [
       {"args": [4], "expected": True},
       {"args": [7], "expected": False},
       {"args": [0], "expected": True},
   ]}, "exec")

_q("code_debug",
   "This function should reverse a string but has a bug:\n"
   "def reverse_string(s):\n    return s\nFind and fix it.",
   {"function_name": "reverse_string", "tests": [
       {"args": ["hello"], "expected": "olleh"},
       {"args": [""], "expected": ""},
       {"args": ["ab"], "expected": "ba"},
   ]}, "exec")

_q("code_debug",
   "This function should compute n factorial but has a bug:\n"
   "def factorial(n):\n    if n == 0:\n        return 0\n"
   "    return n * factorial(n - 1)\nFind and fix it.",
   {"function_name": "factorial", "tests": [
       {"args": [0], "expected": 1},
       {"args": [5], "expected": 120},
       {"args": [1], "expected": 1},
   ]}, "exec")

# ------------------------------------------------------------- code_gen

_q("code_gen",
   "Write a Python function called second_largest that returns the "
   "second-largest number in a list, handling duplicates correctly "
   "(e.g. [5, 5, 3] should return 3).",
   {"function_name": "second_largest", "tests": [
       {"args": [[5, 5, 3]], "expected": 3},
       {"args": [[1, 2, 3, 4]], "expected": 3},
       {"args": [[10, 10, 10, 2]], "expected": 2},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called is_palindrome that returns True if a "
   "given string reads the same forwards and backwards, ignoring case, "
   "and False otherwise.",
   {"function_name": "is_palindrome", "tests": [
       {"args": ["Racecar"], "expected": True},
       {"args": ["hello"], "expected": False},
       {"args": ["level"], "expected": True},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called flatten that takes a nested list of "
   "arbitrary depth and returns a single flat list of all the values, "
   "in order.",
   {"function_name": "flatten", "tests": [
       {"args": [[1, [2, 3], [4, [5, 6]]]], "expected": [1, 2, 3, 4, 5, 6]},
       {"args": [[[1], [2]]], "expected": [1, 2]},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called count_vowels that returns the number "
   "of vowels (a, e, i, o, u, case-insensitive) in a given string.",
   {"function_name": "count_vowels", "tests": [
       {"args": ["Hello World"], "expected": 3},
       {"args": ["xyz"], "expected": 0},
       {"args": ["AEIOU"], "expected": 5},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called nth_fibonacci that returns the n-th "
   "Fibonacci number (0-indexed, so nth_fibonacci(0) == 0 and "
   "nth_fibonacci(1) == 1).",
   {"function_name": "nth_fibonacci", "tests": [
       {"args": [0], "expected": 0},
       {"args": [1], "expected": 1},
       {"args": [10], "expected": 55},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called is_prime that returns True if a given "
   "positive integer is prime, and False otherwise.",
   {"function_name": "is_prime", "tests": [
       {"args": [2], "expected": True},
       {"args": [1], "expected": False},
       {"args": [17], "expected": True},
       {"args": [18], "expected": False},
   ]}, "exec")


def main():
    OUTPUT_PATH.write_text("\n".join(json.dumps(q) for q in QUERIES) + "\n")
    by_category = {}
    for q in QUERIES:
        by_category[q["category"]] = by_category.get(q["category"], 0) + 1
    print(f"wrote {len(QUERIES)} queries to {OUTPUT_PATH}")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat:<14} {count}")


if __name__ == "__main__":
    main()
