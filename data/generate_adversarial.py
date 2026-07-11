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

_q("math",
   "A tank starts with 600 liters. It drains at 10 liters per minute for "
   "12 minutes, then is refilled at 15 liters per minute for 10 minutes, "
   "then drains again at 6 liters per minute for 8 minutes. How many "
   "liters are in the tank now?",
   str(600 - 10 * 12 + 15 * 10 - 6 * 8), "numeric")

_q("math",
   "A store has 500 items. It sells 30% on Monday and 25 more on Tuesday. "
   "How many items remain?",
   str(500 - round(500 * 0.30) - 25), "numeric")

_q("math",
   "A pool starts empty. It's filled at 20 liters per minute for 15 "
   "minutes, then drained at 8 liters per minute for 6 minutes, then "
   "filled at 10 liters per minute for 5 minutes. How many liters are in "
   "the pool now?",
   str(20 * 15 - 8 * 6 + 10 * 5), "numeric")

_q("math",
   "A seller buys widgets for $12 each and sells them for $20 each. After "
   "selling 55 widgets and paying $80 in fixed expenses, what is the "
   "total profit?",
   str((20 - 12) * 55 - 80), "numeric")

_q("math",
   "What is the average of these numbers: 15, 22, 8, 30, 19, 11?",
   str(sum([15, 22, 8, 30, 19, 11]) / 6), "numeric")

_q("math",
   "A price starts at $350. It increases by 15%, then decreases by 20%. "
   "What is the final price?",
   str(350 * 1.15 * 0.8), "numeric")

_q("math",
   "One worker assembles 4 units per hour and works 7 hours. A second "
   "worker assembles 6 units per hour and works 5 hours. How many units "
   "did they assemble in total?",
   str(4 * 7 + 6 * 5), "numeric")

_q("math",
   "An item costs $220. It's discounted 15%, then an additional 10% off "
   "the already-discounted price. What is the final price?",
   str(220 * 0.85 * 0.9), "numeric")

_q("math",
   "A tank starts with 750 liters. It drains at 12 liters per minute for "
   "10 minutes, then is refilled at 18 liters per minute for 15 minutes, "
   "then drains again at 9 liters per minute for 12 minutes. How many "
   "liters are in the tank now?",
   str(750 - 12 * 10 + 18 * 15 - 9 * 12), "numeric")

_q("math",
   "A store has 420 items. It sells 25% on Monday and 40 more on Tuesday. "
   "How many items remain?",
   str(420 - round(420 * 0.25) - 40), "numeric")

_q("math",
   "A pool starts empty. It's filled at 30 liters per minute for 10 "
   "minutes, then drained at 12 liters per minute for 8 minutes, then "
   "filled at 5 liters per minute for 20 minutes. How many liters are in "
   "the pool now?",
   str(30 * 10 - 12 * 8 + 5 * 20), "numeric")

_q("math",
   "A seller buys widgets for $5 each and sells them for $9 each. After "
   "selling 80 widgets and paying $50 in fixed expenses, what is the "
   "total profit?",
   str((9 - 5) * 80 - 50), "numeric")

_q("math",
   "What is the average of these numbers: 4, 16, 9, 25, 6?",
   str(sum([4, 16, 9, 25, 6]) / 5), "numeric")

_q("math",
   "A price starts at $500. It increases by 10%, then decreases by 15%. "
   "What is the final price?",
   str(500 * 1.10 * 0.85), "numeric")

_q("math",
   "One worker assembles 8 units per hour and works 4 hours. A second "
   "worker assembles 2 units per hour and works 12 hours. How many units "
   "did they assemble in total?",
   str(8 * 4 + 2 * 12), "numeric")

_q("math",
   "An item costs $80. It's discounted 20%, then an additional 5% off "
   "the already-discounted price. What is the final price?",
   str(80 * 0.8 * 0.95), "numeric")

_q("math",
   "A tank starts with 900 liters. It drains at 15 liters per minute for "
   "10 minutes, then is refilled at 20 liters per minute for 12 minutes, "
   "then drains again at 10 liters per minute for 6 minutes. How many "
   "liters are in the tank now?",
   str(900 - 15 * 10 + 20 * 12 - 10 * 6), "numeric")

_q("math",
   "A store has 600 items. It sells 35% on Monday and 50 more on Tuesday. "
   "How many items remain?",
   str(600 - round(600 * 0.35) - 50), "numeric")

_q("math",
   "A pool starts empty. It's filled at 18 liters per minute for 20 "
   "minutes, then drained at 9 liters per minute for 10 minutes, then "
   "filled at 6 liters per minute for 15 minutes. How many liters are in "
   "the pool now?",
   str(18 * 20 - 9 * 10 + 6 * 15), "numeric")

_q("math",
   "A seller buys widgets for $7 each and sells them for $13 each. After "
   "selling 65 widgets and paying $70 in fixed expenses, what is the "
   "total profit?",
   str((13 - 7) * 65 - 70), "numeric")

_q("math",
   "What is the average of these numbers: 20, 35, 12, 8, 45?",
   str(sum([20, 35, 12, 8, 45]) / 5), "numeric")

_q("math",
   "A price starts at $280. It increases by 20%, then decreases by 25%. "
   "What is the final price?",
   str(280 * 1.20 * 0.75), "numeric")

_q("math",
   "One worker assembles 6 units per hour and works 8 hours. A second "
   "worker assembles 4 units per hour and works 10 hours. How many units "
   "did they assemble in total?",
   str(6 * 8 + 4 * 10), "numeric")

_q("math",
   "An item costs $95. It's discounted 25%, then an additional 8% off "
   "the already-discounted price. What is the final price?",
   str(95 * 0.75 * 0.92), "numeric")

_q("math",
   "A tank starts with 1000 liters. It drains at 20 liters per minute "
   "for 8 minutes, then is refilled at 25 liters per minute for 6 "
   "minutes, then drains again at 15 liters per minute for 5 minutes. "
   "How many liters are in the tank now?",
   str(1000 - 20 * 8 + 25 * 6 - 15 * 5), "numeric")

_q("math",
   "A store has 800 items. It sells 40% on Monday and 70 more on "
   "Tuesday. How many items remain?",
   str(800 - round(800 * 0.40) - 70), "numeric")

_q("math",
   "A pool starts empty. It's filled at 12 liters per minute for 25 "
   "minutes, then drained at 7 liters per minute for 12 minutes, then "
   "filled at 9 liters per minute for 10 minutes. How many liters are in "
   "the pool now?",
   str(12 * 25 - 7 * 12 + 9 * 10), "numeric")

_q("math",
   "A seller buys widgets for $3 each and sells them for $6 each. After "
   "selling 120 widgets and paying $40 in fixed expenses, what is the "
   "total profit?",
   str((6 - 3) * 120 - 40), "numeric")

_q("math",
   "What is the average of these numbers: 50, 42, 38, 60, 30, 44?",
   str(sum([50, 42, 38, 60, 30, 44]) / 6), "numeric")

_q("math",
   "A price starts at $650. It increases by 8%, then decreases by 12%. "
   "What is the final price?",
   str(650 * 1.08 * 0.88), "numeric")

_q("math",
   "One worker assembles 10 units per hour and works 3 hours. A second "
   "worker assembles 7 units per hour and works 6 hours. How many units "
   "did they assemble in total?",
   str(10 * 3 + 7 * 6), "numeric")

_q("math",
   "An item costs $130. It's discounted 30%, then an additional 10% off "
   "the already-discounted price. What is the final price?",
   str(130 * 0.7 * 0.9), "numeric")

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
    ("The {p} is solid but a bit overpriced for what it offers.", "keyboard", "neutral"),
    ("Absolutely love the {p}, changed my daily routine for the better.", "water bottle", "positive"),
    ("The {p} stopped charging after a week, very disappointing.", "power bank", "negative"),
    ("The {p} works as described, no surprises either way.", "phone case", "neutral"),
    ("Fantastic build quality on this {p}, exceeded what I paid for.", "toolset", "positive"),
    ("The {p} arrived damaged and customer service was unhelpful.", "monitor", "negative"),
    ("It's an average {p}, gets the job done without any flair.", "umbrella", "neutral"),
    ("This {p} is a game changer, I use it every single day now.", "notebook", "positive"),
    ("The {p} is cheaply made and fell apart within a month.", "sunglasses", "negative"),
    ("The {p} matches the photos and specs listed exactly.", "desk mat", "neutral"),
    ("I'm thrilled with this {p}, it's better than the pricier alternatives.", "space heater", "positive"),
    ("Terrible experience, the {p} overheated and shut off on its own.", "hair dryer", "negative"),
    ("The {p} is fine for occasional use, nothing special.", "picnic blanket", "neutral"),
    ("Couldn't be happier with the {p}, it's exactly what I needed.", "tool belt", "positive"),
    ("The {p} misses basic features I expected at this price.", "webcam", "negative"),
    ("The {p} performs consistently, no complaints so far.", "router", "neutral"),
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
    ("Extract all named entities and their types from: Elena Petrova launched a research lab for Quantum Bridge in Warsaw this April.",
     ["Elena Petrova", "Quantum Bridge", "Warsaw", "April"]),
    ("Extract all named entities and their types from: Jonas Berg joined Nordwind Systems as CFO in Stockholm in 2023.",
     ["Jonas Berg", "Nordwind Systems", "Stockholm", "2023"]),
    ("Extract all named entities and their types from: Fatima Zahra opened a design studio for Sable and Co in Casablanca last June.",
     ["Fatima Zahra", "Sable and Co", "Casablanca", "June"]),
    ("Extract all named entities and their types from: Ravi Deshmukh presented the quarterly report for Orion Freight in Mumbai this February.",
     ["Ravi Deshmukh", "Orion Freight", "Mumbai", "February"]),
    ("Extract all named entities and their types from: Clara Jansen relocated to Toronto to head operations at Birchwood Systems in August.",
     ["Clara Jansen", "Toronto", "Birchwood Systems", "August"]),
    ("Extract all named entities and their types from: Tomas Novak signed a licensing deal with Halcyon Robotics in Prague on May 9, 2024.",
     ["Tomas Novak", "Halcyon Robotics", "Prague", "May 9"]),
    ("Extract all named entities and their types from: Grace Muthoni founded Savanna Analytics in Nairobi in 2022.",
     ["Grace Muthoni", "Savanna Analytics", "Nairobi", "2022"]),
    ("Extract all named entities and their types from: Marco Rinaldi joined Vertex Motors as lead engineer in Turin last October.",
     ["Marco Rinaldi", "Vertex Motors", "Turin", "October"]),
    ("Extract all named entities and their types from: Aiko Tanaka presented the new roadmap for Sakura Robotics in Osaka this November.",
     ["Aiko Tanaka", "Sakura Robotics", "Osaka", "November"]),
    ("Extract all named entities and their types from: Diego Fernandez opened a satellite office for Andes Cloud in Bogota in March.",
     ["Diego Fernandez", "Andes Cloud", "Bogota", "March"]),
    ("Extract all named entities and their types from: Ingrid Solberg joined Fjord Energy as director in Bergen in 2021.",
     ["Ingrid Solberg", "Fjord Energy", "Bergen", "2021"]),
    ("Extract all named entities and their types from: Samuel Okoye led the expansion of Delta Freight into Accra this July.",
     ["Samuel Okoye", "Delta Freight", "Accra", "July"]),
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
    ("The transit authority confirmed that the downtown subway extension "
     "will open two months ahead of schedule, following faster-than-"
     "expected tunnel construction.",
     ["subway", "ahead of schedule", "tunnel"]),
    ("A regional hospital network announced a partnership with a "
     "telehealth startup to provide free virtual consultations to rural "
     "patients starting next spring.",
     ["hospital", "telehealth", "rural"]),
    ("The national park service reported that wolf reintroduction efforts "
     "have led to a marked recovery in local elk and vegetation "
     "populations over the past decade.",
     ["wolf", "elk", "vegetation"]),
    ("A mid-sized furniture manufacturer is shifting its entire production "
     "line to reclaimed wood, citing both cost savings and demand from "
     "environmentally conscious buyers.",
     ["furniture", "reclaimed wood", "demand"]),
    ("City officials unveiled a plan to convert an abandoned rail line "
     "into a fifteen-kilometre pedestrian and cycling greenway, with the "
     "first section opening next summer.",
     ["rail line", "greenway", "cycling"]),
    ("Researchers found that a newly discovered species of deep-sea coral "
     "can survive water temperatures previously thought lethal to most "
     "reef-building organisms.",
     ["coral", "deep-sea", "temperatures"]),
    ("The school district approved funding for a free breakfast program "
     "in all elementary schools after a pilot showed improved attendance "
     "and test scores.",
     ["breakfast", "elementary", "attendance"]),
    ("A logistics company is testing electric delivery bikes in three "
     "congested city centers, reporting a thirty percent drop in average "
     "delivery times so far.",
     ["electric", "delivery bikes", "delivery times"]),
    ("The museum announced it will digitize its entire photo archive over "
     "the next three years, making over two hundred thousand historical "
     "images publicly searchable online.",
     ["digitize", "photo archive", "historical"]),
    ("A community farming cooperative doubled its harvest yield this year "
     "after switching to a crop rotation system recommended by a local "
     "agricultural extension office.",
     ["farming", "harvest", "crop rotation"]),
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
    ("What is the capital of Japan?", ["Tokyo"]),
    ("What is the chemical symbol for gold?", ["Au"]),
    ("What planet is known as the Red Planet?", ["Mars"]),
    ("Who wrote the play Romeo and Juliet?", ["Shakespeare"]),
    ("What is the largest ocean on Earth?", ["Pacific"]),
    ("What gas do plants primarily absorb during photosynthesis?", ["carbon dioxide"]),
    ("What is the freezing point of water in Celsius?", ["0"]),
    ("Who painted the Mona Lisa?", ["da Vinci"]),
    ("What is the smallest prime number?", ["2"]),
    ("What country is home to the Great Barrier Reef?", ["Australia"]),
    ("What is the official currency of Japan called?", ["yen"]),
    ("What is the tallest mountain in the world?", ["Everest"]),
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

_h = _solve_unique(
    ["Zara", "Finn", "Talia"], ["tea", "soda", "juice"],
    [lambda a: a["Zara"] != "soda", lambda a: a["Finn"] == "juice"],
)
_q("logic",
   "Three friends, Zara, Finn, and Talia, each drink a different "
   "beverage: tea, soda, juice. Zara doesn't drink soda. Finn drinks "
   "juice. Who drinks tea?",
   [n for n, v in _h.items() if v == "tea"][0], "exact")

_i = _solve_unique(
    ["Omar", "Bea", "Nils"], ["chess", "checkers", "dominoes"],
    [lambda a: a["Omar"] != "dominoes", lambda a: a["Bea"] == "checkers"],
)
_q("logic",
   "Three friends, Omar, Bea, and Nils, each play a different game: "
   "chess, checkers, dominoes. Omar doesn't play dominoes. Bea plays "
   "checkers. Who plays dominoes?",
   [n for n, v in _i.items() if v == "dominoes"][0], "exact")

_j = _solve_unique(
    ["Petra", "Junho", "Alexei"], ["hiking", "swimming", "cycling"],
    [lambda a: a["Petra"] != "cycling", lambda a: a["Junho"] == "swimming"],
)
_q("logic",
   "Three friends, Petra, Junho, and Alexei, each prefer a different "
   "activity: hiking, swimming, cycling. Petra doesn't prefer cycling. "
   "Junho prefers swimming. Who prefers cycling?",
   [n for n, v in _j.items() if v == "cycling"][0], "exact")

_k = _solve_unique(
    ["Ines", "Cormac", "Wren"], ["cake", "pie", "tart"],
    [lambda a: a["Ines"] != "tart", lambda a: a["Cormac"] == "pie"],
)
_q("logic",
   "Three friends, Ines, Cormac, and Wren, each ordered a different "
   "dessert: cake, pie, tart. Ines didn't order the tart. Cormac ordered "
   "pie. Who ordered the tart?",
   [n for n, v in _k.items() if v == "tart"][0], "exact")

_l = _solve_unique(
    ["Soren", "Amara", "Dax"], ["guitar", "drums", "bass"],
    [lambda a: a["Soren"] != "bass", lambda a: a["Amara"] == "drums"],
)
_q("logic",
   "Three musicians, Soren, Amara, and Dax, each play a different "
   "instrument: guitar, drums, bass. Soren doesn't play bass. Amara "
   "plays drums. Who plays bass?",
   [n for n, v in _l.items() if v == "bass"][0], "exact")

_m = _solve_unique(
    ["Yara", "Bram", "Isla"], ["sedan", "truck", "van"],
    [lambda a: a["Yara"] != "van", lambda a: a["Bram"] == "truck"],
)
_q("logic",
   "Three friends, Yara, Bram, and Isla, each drive a different vehicle: "
   "sedan, truck, van. Yara doesn't drive the van. Bram drives the "
   "truck. Who drives the van?",
   [n for n, v in _m.items() if v == "van"][0], "exact")

_n = _solve_unique(
    ["Milo", "Reza", "Ffion"], ["soccer", "baseball", "hockey"],
    [lambda a: a["Milo"] != "hockey", lambda a: a["Reza"] == "baseball"],
)
_q("logic",
   "Three friends, Milo, Reza, and Ffion, each play a different sport: "
   "soccer, baseball, hockey. Milo doesn't play hockey. Reza plays "
   "baseball. Who plays hockey?",
   [n for n, v in _n.items() if v == "hockey"][0], "exact")

_o = _solve_unique(
    ["Petra", "Doran", "Liv", "Amos"], ["pasta", "salad", "soup", "sandwich"],
    [lambda a: a["Petra"] != "pasta", lambda a: a["Petra"] != "sandwich",
     lambda a: a["Doran"] == "salad", lambda a: a["Liv"] != "sandwich"],
)
_q("logic",
   "Four friends, Petra, Doran, Liv, and Amos, each ordered a different "
   "dish: pasta, salad, soup, sandwich. Petra didn't order pasta or the "
   "sandwich. Doran ordered salad. Liv didn't order the sandwich. Who "
   "ordered the sandwich?",
   [n for n, v in _o.items() if v == "sandwich"][0], "exact")

_p = _solve_unique(
    ["Enzo", "Ruth", "Kwame", "Sena"], ["math", "science", "art", "music"],
    [lambda a: a["Enzo"] != "art", lambda a: a["Enzo"] != "music",
     lambda a: a["Ruth"] == "science", lambda a: a["Kwame"] != "music"],
)
_q("logic",
   "Four students, Enzo, Ruth, Kwame, and Sena, each study a different "
   "subject: math, science, art, music. Enzo doesn't study art or "
   "music. Ruth studies science. Kwame doesn't study music. Who studies "
   "music?",
   [n for n, v in _p.items() if v == "music"][0], "exact")

_r = _solve_unique(
    ["Odette", "Hassan", "Ines", "Bo"], ["spring", "summer", "autumn", "winter"],
    [lambda a: a["Odette"] != "summer", lambda a: a["Odette"] != "winter",
     lambda a: a["Hassan"] == "autumn", lambda a: a["Ines"] != "winter"],
)
_q("logic",
   "Four friends, Odette, Hassan, Ines, and Bo, each have a different "
   "favorite season: spring, summer, autumn, winter. Odette's favorite "
   "isn't summer or winter. Hassan's favorite is autumn. Ines's favorite "
   "isn't winter. Whose favorite season is winter?",
   [n for n, v in _r.items() if v == "winter"][0], "exact")

_s = _solve_unique(
    ["Farah", "Otis", "Maren", "Cato"], ["blue", "silver", "black", "white"],
    [lambda a: a["Farah"] != "black", lambda a: a["Farah"] != "white",
     lambda a: a["Otis"] == "silver", lambda a: a["Maren"] != "white"],
)
_q("logic",
   "Four friends, Farah, Otis, Maren, and Cato, each have a different "
   "colored phone: blue, silver, black, white. Farah's phone isn't black "
   "or white. Otis's phone is silver. Maren's phone isn't white. Whose "
   "phone is white?",
   [n for n, v in _s.items() if v == "white"][0], "exact")

_t = _solve_unique(
    ["Idris", "Noelle", "Petros", "Wyn"], ["north", "south", "east", "west"],
    [lambda a: a["Idris"] != "east", lambda a: a["Idris"] != "west",
     lambda a: a["Noelle"] == "south", lambda a: a["Petros"] != "west"],
)
_q("logic",
   "Four coworkers, Idris, Noelle, Petros, and Wyn, were each assigned a "
   "different region: north, south, east, west. Idris wasn't assigned "
   "east or west. Noelle was assigned south. Petros wasn't assigned "
   "west. Who was assigned west?",
   [n for n, v in _t.items() if v == "west"][0], "exact")

_u = _solve_unique(
    ["Beatrix", "Njeri", "Osman", "Tamsin"], ["monday", "tuesday", "wednesday", "thursday"],
    [lambda a: a["Beatrix"] != "wednesday", lambda a: a["Beatrix"] != "thursday",
     lambda a: a["Njeri"] == "tuesday", lambda a: a["Osman"] != "thursday"],
)
_q("logic",
   "Four coworkers, Beatrix, Njeri, Osman, and Tamsin, each work a "
   "different shift day: Monday, Tuesday, Wednesday, Thursday. Beatrix "
   "doesn't work Wednesday or Thursday. Njeri works Tuesday. Osman "
   "doesn't work Thursday. Who works Thursday?",
   [n for n, v in _u.items() if v == "thursday"][0], "exact")

_v = _solve_unique(
    ["Quinn", "Radu", "Selin", "Toby"], ["hiking-boots", "sandals", "sneakers", "slippers"],
    [lambda a: a["Quinn"] != "sneakers", lambda a: a["Quinn"] != "slippers",
     lambda a: a["Radu"] == "sandals", lambda a: a["Selin"] != "slippers"],
)
_q("logic",
   "Four friends, Quinn, Radu, Selin, and Toby, each prefer different "
   "footwear: hiking boots, sandals, sneakers, slippers. Quinn doesn't "
   "prefer sneakers or slippers. Radu prefers sandals. Selin doesn't "
   "prefer slippers. Who prefers slippers?",
   [n for n, v in _v.items() if v == "slippers"][0], "exact")

_w = _solve_unique(
    ["Amina", "Luca", "Soraya"], ["bread", "rice", "pasta"],
    [lambda a: a["Amina"] != "pasta", lambda a: a["Luca"] == "rice"],
)
_q("logic",
   "Three friends, Amina, Luca, and Soraya, each ate a different staple "
   "food: bread, rice, pasta. Amina didn't eat pasta. Luca ate rice. Who "
   "ate pasta?",
   [n for n, v in _w.items() if v == "pasta"][0], "exact")

_x = _solve_unique(
    ["Denis", "Priya", "Oskar"], ["novel", "comic", "magazine"],
    [lambda a: a["Denis"] != "magazine", lambda a: a["Priya"] == "comic"],
)
_q("logic",
   "Three friends, Denis, Priya, and Oskar, each are reading a different "
   "kind of publication: novel, comic, magazine. Denis isn't reading the "
   "magazine. Priya is reading a comic. Who is reading the magazine?",
   [n for n, v in _x.items() if v == "magazine"][0], "exact")

_y = _solve_unique(
    ["Mireille", "Kenji", "Abel"], ["painting", "sketching", "sculpture"],
    [lambda a: a["Mireille"] != "sculpture", lambda a: a["Kenji"] == "sketching"],
)
_q("logic",
   "Three artists, Mireille, Kenji, and Abel, each work in a different "
   "medium: painting, sketching, sculpture. Mireille doesn't work in "
   "sculpture. Kenji works in sketching. Who works in sculpture?",
   [n for n, v in _y.items() if v == "sculpture"][0], "exact")

_z = _solve_unique(
    ["Halle", "Ronan", "Xia"], ["apple juice", "orange juice", "grape juice"],
    [lambda a: a["Halle"] != "grape juice", lambda a: a["Ronan"] == "orange juice"],
)
_q("logic",
   "Three friends, Halle, Ronan, and Xia, each drink a different juice: "
   "apple, orange, grape. Halle doesn't drink grape juice. Ronan drinks "
   "orange juice. Who drinks grape juice?",
   [n for n, v in _z.items() if v == "grape juice"][0], "exact")

_aa = _solve_unique(
    ["Tobias", "Nadia", "Emre"], ["hockey", "tennis", "badminton"],
    [lambda a: a["Tobias"] != "badminton", lambda a: a["Nadia"] == "tennis"],
)
_q("logic",
   "Three friends, Tobias, Nadia, and Emre, each play a different sport: "
   "hockey, tennis, badminton. Tobias doesn't play badminton. Nadia "
   "plays tennis. Who plays badminton?",
   [n for n, v in _aa.items() if v == "badminton"][0], "exact")

_ab = _solve_unique(
    ["Ingrid", "Pavel", "Chidi"], ["rose", "tulip", "daisy"],
    [lambda a: a["Ingrid"] != "daisy", lambda a: a["Pavel"] == "tulip"],
)
_q("logic",
   "Three gardeners, Ingrid, Pavel, and Chidi, each grow a different "
   "flower: rose, tulip, daisy. Ingrid doesn't grow daisies. Pavel grows "
   "tulips. Who grows daisies?",
   [n for n, v in _ab.items() if v == "daisy"][0], "exact")

_ac = _solve_unique(
    ["Saoirse", "Dmitri", "Lina"], ["bus", "train", "bike"],
    [lambda a: a["Saoirse"] != "bike", lambda a: a["Dmitri"] == "train"],
)
_q("logic",
   "Three friends, Saoirse, Dmitri, and Lina, each commute a different "
   "way: bus, train, bike. Saoirse doesn't commute by bike. Dmitri "
   "commutes by train. Who commutes by bike?",
   [n for n, v in _ac.items() if v == "bike"][0], "exact")

_ad = _solve_unique(
    ["Yusuf", "Camille", "Bashir", "Freya"], ["red wine", "white wine", "cider", "beer"],
    [lambda a: a["Yusuf"] != "cider", lambda a: a["Yusuf"] != "beer",
     lambda a: a["Camille"] == "white wine", lambda a: a["Bashir"] != "beer"],
)
_q("logic",
   "Four friends, Yusuf, Camille, Bashir, and Freya, each ordered a "
   "different drink: red wine, white wine, cider, beer. Yusuf didn't "
   "order cider or beer. Camille ordered white wine. Bashir didn't order "
   "beer. Who ordered beer?",
   [n for n, v in _ad.items() if v == "beer"][0], "exact")

_ae = _solve_unique(
    ["Arun", "Zainab", "Peder", "Mila"], ["north", "south", "east", "west"],
    [lambda a: a["Arun"] != "east", lambda a: a["Arun"] != "west",
     lambda a: a["Zainab"] == "south", lambda a: a["Peder"] != "west"],
)
_q("logic",
   "Four coworkers, Arun, Zainab, Peder, and Mila, were each assigned a "
   "different office wing: north, south, east, west. Arun wasn't "
   "assigned east or west. Zainab was assigned south. Peder wasn't "
   "assigned west. Who was assigned the west wing?",
   [n for n, v in _ae.items() if v == "west"][0], "exact")

_af = _solve_unique(
    ["Katarina", "Boaz", "Ifeoma", "Stellan"], ["cello", "harp", "oboe", "clarinet"],
    [lambda a: a["Katarina"] != "oboe", lambda a: a["Katarina"] != "clarinet",
     lambda a: a["Boaz"] == "harp", lambda a: a["Ifeoma"] != "clarinet"],
)
_q("logic",
   "Four musicians, Katarina, Boaz, Ifeoma, and Stellan, each play a "
   "different instrument: cello, harp, oboe, clarinet. Katarina doesn't "
   "play oboe or clarinet. Boaz plays harp. Ifeoma doesn't play "
   "clarinet. Who plays clarinet?",
   [n for n, v in _af.items() if v == "clarinet"][0], "exact")

_ag = _solve_unique(
    ["Rosalind", "Achille", "Ngozi", "Baptiste"], ["gold medal", "silver medal", "bronze medal", "ribbon"],
    [lambda a: a["Rosalind"] != "bronze medal", lambda a: a["Rosalind"] != "ribbon",
     lambda a: a["Achille"] == "silver medal", lambda a: a["Ngozi"] != "ribbon"],
)
_q("logic",
   "Four competitors, Rosalind, Achille, Ngozi, and Baptiste, each "
   "received a different award: gold medal, silver medal, bronze medal, "
   "ribbon. Rosalind didn't receive the bronze medal or the ribbon. "
   "Achille received the silver medal. Ngozi didn't receive the ribbon. "
   "Who received the ribbon?",
   [n for n, v in _ag.items() if v == "ribbon"][0], "exact")

_ah = _solve_unique(
    ["Esben", "Ayaan", "Marguerite", "Ochieng"], ["python", "java", "rust", "go"],
    [lambda a: a["Esben"] != "rust", lambda a: a["Esben"] != "go",
     lambda a: a["Ayaan"] == "java", lambda a: a["Marguerite"] != "go"],
)
_q("logic",
   "Four developers, Esben, Ayaan, Marguerite, and Ochieng, each write "
   "in a different programming language: Python, Java, Rust, Go. Esben "
   "doesn't write Rust or Go. Ayaan writes Java. Marguerite doesn't "
   "write Go. Who writes Go?",
   [n for n, v in _ah.items() if v == "go"][0], "exact")

_ai = _solve_unique(
    ["Perpetua", "Njord", "Aziza", "Colm"], ["morning shift", "afternoon shift", "evening shift", "night shift"],
    [lambda a: a["Perpetua"] != "evening shift", lambda a: a["Perpetua"] != "night shift",
     lambda a: a["Njord"] == "afternoon shift", lambda a: a["Aziza"] != "night shift"],
)
_q("logic",
   "Four coworkers, Perpetua, Njord, Aziza, and Colm, each work a "
   "different shift: morning, afternoon, evening, night. Perpetua "
   "doesn't work evening or night. Njord works afternoon. Aziza doesn't "
   "work night. Who works the night shift?",
   [n for n, v in _ai.items() if v == "night shift"][0], "exact")

_aj = _solve_unique(
    ["Wilhelmina", "Boyd", "Nkechi", "Tobias"], ["strawberry", "vanilla", "chocolate", "mint"],
    [lambda a: a["Wilhelmina"] != "chocolate", lambda a: a["Wilhelmina"] != "mint",
     lambda a: a["Boyd"] == "vanilla", lambda a: a["Nkechi"] != "mint"],
)
_q("logic",
   "Four friends, Wilhelmina, Boyd, Nkechi, and Tobias, each chose a "
   "different ice cream flavor: strawberry, vanilla, chocolate, mint. "
   "Wilhelmina didn't choose chocolate or mint. Boyd chose vanilla. "
   "Nkechi didn't choose mint. Who chose mint?",
   [n for n, v in _aj.items() if v == "mint"][0], "exact")

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

_q("code_debug",
   "This function should sum the even numbers in a list but has a bug:\n"
   "def sum_evens(nums):\n    total = 0\n"
   "    for i in range(len(nums) - 1):\n"
   "        if nums[i] % 2 == 0:\n            total += nums[i]\n"
   "    return total\nFind and fix it.",
   {"function_name": "sum_evens", "tests": [
       {"args": [[1, 2, 3, 4]], "expected": 6},
       {"args": [[2, 4, 6]], "expected": 12},
       {"args": [[1, 3, 5]], "expected": 0},
   ]}, "exec")

_q("code_debug",
   "This function should return whether someone is an adult (18 or "
   "older) but has a bug:\ndef is_adult(age):\n    return age > 18\n"
   "Find and fix it.",
   {"function_name": "is_adult", "tests": [
       {"args": [18], "expected": True},
       {"args": [17], "expected": False},
       {"args": [19], "expected": True},
   ]}, "exec")

_q("code_debug",
   "This function should return the smallest number in a list but has a "
   "bug:\ndef find_min(nums):\n    m = 0\n    for n in nums:\n"
   "        if n < m:\n            m = n\n    return m\nFind and fix it.",
   {"function_name": "find_min", "tests": [
       {"args": [[5, 3, 8, 1]], "expected": 1},
       {"args": [[10, 20, 30]], "expected": 10},
       {"args": [[-5, -1, -9]], "expected": -9},
   ]}, "exec")

_q("code_debug",
   "This function should title-case every word in a string but has a "
   "bug:\ndef capitalize_words(s):\n    return s.lower()\n"
   "Find and fix it.",
   {"function_name": "capitalize_words", "tests": [
       {"args": ["hello world"], "expected": "Hello World"},
       {"args": ["PYTHON"], "expected": "Python"},
       {"args": ["a b c"], "expected": "A B C"},
   ]}, "exec")

_q("code_debug",
   "This function should count how many times a value appears in a list "
   "but has a bug:\ndef count_occurrences(items, target):\n"
   "    count = 0\n    for item in items:\n        if item == target:\n"
   "            count = 1\n    return count\nFind and fix it.",
   {"function_name": "count_occurrences", "tests": [
       {"args": [[1, 2, 1, 1, 3], 1], "expected": 3},
       {"args": [["x", "y"], "z"], "expected": 0},
       {"args": [[5, 5, 5], 5], "expected": 3},
   ]}, "exec")

_q("code_debug",
   "This function should sum the digits of a number but has a bug:\n"
   "def digit_sum(n):\n    total = 0\n    n = abs(n)\n"
   "    while n > 0:\n        total += n % 10\n        n = n // 100\n"
   "    return total\nFind and fix it.",
   {"function_name": "digit_sum", "tests": [
       {"args": [123], "expected": 6},
       {"args": [9], "expected": 9},
       {"args": [-45], "expected": 9},
   ]}, "exec")

_q("code_debug",
   "This function should remove duplicates from a list while keeping "
   "first-occurrence order but has a bug:\n"
   "def dedupe(items):\n    return list(set(items))\nFind and fix it.",
   {"function_name": "dedupe", "tests": [
       {"args": [[5, 3, 3, 5, 1]], "expected": [5, 3, 1]},
       {"args": [[9, 2, 9, 7]], "expected": [9, 2, 7]},
       {"args": [[4, 4, 4, 4]], "expected": [4]},
   ]}, "exec")

_q("code_debug",
   "This function should check if two strings are anagrams of each "
   "other, ignoring case, but has a bug:\n"
   "def is_anagram(a, b):\n    return sorted(a) == sorted(b)\n"
   "Find and fix it.",
   {"function_name": "is_anagram", "tests": [
       {"args": ["Listen", "silent"], "expected": True},
       {"args": ["abc", "abd"], "expected": False},
       {"args": ["Dormitory", "dirtyroom"], "expected": True},
   ]}, "exec")

_q("code_debug",
   "This function should return the running (cumulative) total before "
   "each number is added but has a bug:\n"
   "def running_total(nums):\n    result = []\n    total = 0\n"
   "    for n in nums:\n        result.append(total)\n        total += n\n"
   "    return result\nIt should instead return the cumulative total "
   "after each number is added. Find and fix it.",
   {"function_name": "running_total", "tests": [
       {"args": [[1, 2, 3]], "expected": [1, 3, 6]},
       {"args": [[5, 5, 5]], "expected": [5, 10, 15]},
       {"args": [[2]], "expected": [2]},
   ]}, "exec")

_q("code_debug",
   "This function should return whether a list is sorted in ascending "
   "order but has a bug:\ndef is_sorted(nums):\n"
   "    for i in range(len(nums) - 1):\n"
   "        if nums[i] > nums[i + 1]:\n            return True\n"
   "    return False\nFind and fix it.",
   {"function_name": "is_sorted", "tests": [
       {"args": [[1, 2, 3]], "expected": True},
       {"args": [[3, 1, 2]], "expected": False},
       {"args": [[5]], "expected": True},
   ]}, "exec")

_q("code_debug",
   "This function should return whether a string is empty but has a "
   "bug:\ndef is_empty(s):\n    return s == None\nFind and fix it.",
   {"function_name": "is_empty", "tests": [
       {"args": [""], "expected": True},
       {"args": ["a"], "expected": False},
       {"args": [" "], "expected": False},
   ]}, "exec")

_q("code_debug",
   "This function should return the product of all numbers in a list "
   "but has a bug:\ndef product(nums):\n    result = 0\n"
   "    for n in nums:\n        result *= n\n    return result\n"
   "Find and fix it.",
   {"function_name": "product", "tests": [
       {"args": [[1, 2, 3, 4]], "expected": 24},
       {"args": [[5]], "expected": 5},
       {"args": [[2, 2, 2]], "expected": 8},
   ]}, "exec")

_q("code_debug",
   "This function should return whether a string contains only digits "
   "but has a bug:\ndef is_numeric(s):\n    return s.isalpha()\n"
   "Find and fix it.",
   {"function_name": "is_numeric", "tests": [
       {"args": ["123"], "expected": True},
       {"args": ["12a"], "expected": False},
       {"args": ["abc"], "expected": False},
   ]}, "exec")

_q("code_debug",
   "This function should merge two already-sorted lists into one sorted "
   "(ascending) list but has a bug:\n"
   "def merge_sorted(a, b):\n    return sorted(a + b, reverse=True)\n"
   "Find and fix it.",
   {"function_name": "merge_sorted", "tests": [
       {"args": [[1, 3, 5], [2, 4, 6]], "expected": [1, 2, 3, 4, 5, 6]},
       {"args": [[], [1, 2]], "expected": [1, 2]},
       {"args": [[5], []], "expected": [5]},
   ]}, "exec")

_q("code_debug",
   "This function should check if parentheses in a string are balanced "
   "but has a bug:\ndef is_balanced(s):\n    count = 0\n"
   "    for c in s:\n        if c == '(':\n            count += 1\n"
   "        elif c == ')':\n            count -= 1\n    return True\n"
   "Find and fix it.",
   {"function_name": "is_balanced", "tests": [
       {"args": ["(())"], "expected": True},
       {"args": ["(()"], "expected": False},
       {"args": [")("], "expected": False},
   ]}, "exec")

_q("code_debug",
   "This function should compute the average of a list excluding one "
   "given value (only the first match, if it appears) but has a bug - "
   "it isn't excluding anything at all:\n"
   "def average_excluding(nums, exclude):\n    total = sum(nums)\n"
   "    count = len(nums)\n    return total / count\nFind and fix it.",
   {"function_name": "average_excluding", "tests": [
       {"args": [[10, 20, 30, 40], 10], "expected": sum([20, 30, 40]) / 3},
       {"args": [[5, 15, 25, 35], 35], "expected": sum([5, 15, 25]) / 3},
       {"args": [[100, 200, 300, 400], 200], "expected": sum([100, 300, 400]) / 3},
   ]}, "exec")

_q("code_debug",
   "This function should return whether every element in a list is "
   "unique but has inverted logic:\n"
   "def all_unique(items):\n    seen = []\n    for i in items:\n"
   "        if i in seen:\n            return True\n"
   "        seen.append(i)\n    return False\nFind and fix it.",
   {"function_name": "all_unique", "tests": [
       {"args": [[1, 2, 3]], "expected": True},
       {"args": [[1, 2, 2]], "expected": False},
       {"args": [[]], "expected": True},
   ]}, "exec")

_q("code_debug",
   "This function should convert a list of (key, value) tuples into a "
   "dict but has a bug:\n"
   "def pairs_to_dict(pairs):\n    return {pairs[0]: pairs[1]}\n"
   "Find and fix it.",
   {"function_name": "pairs_to_dict", "tests": [
       {"args": [[["a", 1], ["b", 2]]], "expected": {"a": 1, "b": 2}},
       {"args": [[["x", 10]]], "expected": {"x": 10}},
       {"args": [[]], "expected": {}},
   ]}, "exec")

_q("code_debug",
   "This function should compute n factorial iteratively but has a "
   "bug:\ndef factorial_iter(n):\n    result = 1\n"
   "    for i in range(1, n):\n        result *= i\n    return result\n"
   "Find and fix it.",
   {"function_name": "factorial_iter", "tests": [
       {"args": [5], "expected": 120},
       {"args": [1], "expected": 1},
       {"args": [0], "expected": 1},
   ]}, "exec")

_q("code_debug",
   "This function should check if n is a multiple of a given number but "
   "has a bug:\ndef is_multiple(n, of):\n    return n % of == 1\n"
   "Find and fix it.",
   {"function_name": "is_multiple", "tests": [
       {"args": [10, 5], "expected": True},
       {"args": [10, 3], "expected": False},
       {"args": [0, 5], "expected": True},
   ]}, "exec")

_q("code_debug",
   "This function should remove all vowels from a string but has "
   "inverted logic:\n"
   "def remove_vowels(s):\n    return \"\".join(c for c in s if c in "
   "\"aeiou\")\nFind and fix it.",
   {"function_name": "remove_vowels", "tests": [
       {"args": ["hello"], "expected": "hll"},
       {"args": ["xyz"], "expected": "xyz"},
       {"args": ["banana"], "expected": "bnn"},
   ]}, "exec")

_q("code_debug",
   "This function should return the range (max minus min) of a list "
   "but has a bug:\ndef value_range(nums):\n"
   "    return max(nums) + min(nums)\nFind and fix it.",
   {"function_name": "value_range", "tests": [
       {"args": [[1, 5, 3]], "expected": 4},
       {"args": [[10, 10, 10]], "expected": 0},
       {"args": [[-5, 5]], "expected": 10},
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

_q("code_gen",
   "Write a Python function called gcd that returns the greatest common "
   "divisor of two non-negative integers.",
   {"function_name": "gcd", "tests": [
       {"args": [12, 18], "expected": 6},
       {"args": [17, 5], "expected": 1},
       {"args": [0, 5], "expected": 5},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called remove_duplicates that returns a list "
   "with duplicate values removed, keeping only the first occurrence of "
   "each value in its original order.",
   {"function_name": "remove_duplicates", "tests": [
       {"args": [[1, 2, 2, 3, 1]], "expected": [1, 2, 3]},
       {"args": [[5, 5, 5]], "expected": [5]},
       {"args": [[]], "expected": []},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called title_case that returns a copy of a "
   "string with the first letter of every word capitalized and the rest "
   "lowercase.",
   {"function_name": "title_case", "tests": [
       {"args": ["the quick fox"], "expected": "The Quick Fox"},
       {"args": ["HELLO"], "expected": "Hello"},
       {"args": ["a"], "expected": "A"},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called sum_of_digits that returns the sum of "
   "the digits of the absolute value of a given integer.",
   {"function_name": "sum_of_digits", "tests": [
       {"args": [1234], "expected": 10},
       {"args": [0], "expected": 0},
       {"args": [-56], "expected": 11},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called is_leap_year that returns True if a "
   "given year is a leap year (divisible by 4, except centuries which "
   "must be divisible by 400), and False otherwise.",
   {"function_name": "is_leap_year", "tests": [
       {"args": [2024], "expected": True},
       {"args": [1900], "expected": False},
       {"args": [2000], "expected": True},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called reverse_words that takes a sentence "
   "and returns it with the order of its words reversed, keeping each "
   "word itself unchanged.",
   {"function_name": "reverse_words", "tests": [
       {"args": ["hello world"], "expected": "world hello"},
       {"args": ["a b c"], "expected": "c b a"},
       {"args": ["python"], "expected": "python"},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called most_common_char that returns the "
   "character that appears most often in a string (you can assume there "
   "is always a single unique most-frequent character in the test "
   "inputs).",
   {"function_name": "most_common_char", "tests": [
       {"args": ["aabbbcc"], "expected": "b"},
       {"args": ["xxyz"], "expected": "x"},
       {"args": ["wwwyyyyz"], "expected": "y"},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called is_perfect_square that returns True "
   "if a given non-negative integer is a perfect square, and False "
   "otherwise.",
   {"function_name": "is_perfect_square", "tests": [
       {"args": [16], "expected": True},
       {"args": [15], "expected": False},
       {"args": [0], "expected": True},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called rotate_list that rotates a list to "
   "the left by k positions (k may be larger than the list's length).",
   {"function_name": "rotate_list", "tests": [
       {"args": [[1, 2, 3, 4, 5], 2], "expected": [3, 4, 5, 1, 2]},
       {"args": [[1, 2, 3], 0], "expected": [1, 2, 3]},
       {"args": [[1, 2], 5], "expected": [2, 1]},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called word_count that returns the number "
   "of words in a string, treating any run of whitespace as a "
   "separator.",
   {"function_name": "word_count", "tests": [
       {"args": ["hello world foo"], "expected": 3},
       {"args": [""], "expected": 0},
       {"args": ["  spaced   out  "], "expected": 2},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called is_armstrong_number that returns True "
   "if a given positive integer equals the sum of its own digits each "
   "raised to the power of the number of digits, and False otherwise.",
   {"function_name": "is_armstrong_number", "tests": [
       {"args": [153], "expected": True},
       {"args": [10], "expected": False},
       {"args": [9474], "expected": True},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called caesar_cipher_encode that shifts "
   "every letter in a string forward by a given number of positions in "
   "the alphabet, wrapping around from z to a, preserving each letter's "
   "case, and leaving non-letter characters unchanged.",
   {"function_name": "caesar_cipher_encode", "tests": [
       {"args": ["abc", 1], "expected": "bcd"},
       {"args": ["xyz", 3], "expected": "abc"},
       {"args": ["Hello, World!", 5], "expected": "Mjqqt, Btwqi!"},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called binary_to_decimal that converts a "
   "string of 0s and 1s into its decimal integer value.",
   {"function_name": "binary_to_decimal", "tests": [
       {"args": ["101"], "expected": 5},
       {"args": ["0"], "expected": 0},
       {"args": ["1111"], "expected": 15},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called decimal_to_binary that converts a "
   "non-negative integer into its binary string representation, with no "
   "leading '0b' prefix.",
   {"function_name": "decimal_to_binary", "tests": [
       {"args": [5], "expected": "101"},
       {"args": [0], "expected": "0"},
       {"args": [255], "expected": "11111111"},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called longest_word that returns the "
   "longest word in a sentence (you can assume there's always a single "
   "unique longest word in the test inputs).",
   {"function_name": "longest_word", "tests": [
       {"args": ["I love programming"], "expected": "programming"},
       {"args": ["cat dog elephant"], "expected": "elephant"},
       {"args": ["a bb ccc"], "expected": "ccc"},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called matrix_transpose that returns the "
   "transpose of a 2D list (a list of lists, all rows the same length).",
   {"function_name": "matrix_transpose", "tests": [
       {"args": [[[1, 2], [3, 4]]], "expected": [[1, 3], [2, 4]]},
       {"args": [[[1, 2, 3]]], "expected": [[1], [2], [3]]},
       {"args": [[[1], [2], [3]]], "expected": [[1, 2, 3]]},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called celsius_to_fahrenheit that converts "
   "a temperature in Celsius to Fahrenheit.",
   {"function_name": "celsius_to_fahrenheit", "tests": [
       {"args": [0], "expected": 32},
       {"args": [100], "expected": 212},
       {"args": [-40], "expected": -40},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called is_pangram that returns True if a "
   "string contains every letter of the English alphabet at least once, "
   "ignoring case, and False otherwise.",
   {"function_name": "is_pangram", "tests": [
       {"args": ["The quick brown fox jumps over the lazy dog"], "expected": True},
       {"args": ["hello world"], "expected": False},
       {"args": ["Pack my box with five dozen liquor jugs"], "expected": True},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called chunk_list that splits a list into "
   "consecutive chunks of a given size, where the last chunk may be "
   "smaller than the rest.",
   {"function_name": "chunk_list", "tests": [
       {"args": [[1, 2, 3, 4, 5], 2], "expected": [[1, 2], [3, 4], [5]]},
       {"args": [[1, 2, 3], 5], "expected": [[1, 2, 3]]},
       {"args": [[], 3], "expected": []},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called first_unique_char that returns the "
   "first character in a string that doesn't repeat anywhere else in "
   "it, or None if every character repeats.",
   {"function_name": "first_unique_char", "tests": [
       {"args": ["swiss"], "expected": "w"},
       {"args": ["aabbcc"], "expected": None},
       {"args": ["teeter"], "expected": "r"},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called is_subsequence that returns True if "
   "every character of string a appears in string b in the same order "
   "(not necessarily contiguously), and False otherwise.",
   {"function_name": "is_subsequence", "tests": [
       {"args": ["ace", "abcde"], "expected": True},
       {"args": ["aec", "abcde"], "expected": False},
       {"args": ["", "abc"], "expected": True},
   ]}, "exec")

_q("code_gen",
   "Write a Python function called max_subarray_sum that returns the "
   "largest possible sum of any contiguous subarray of a list of "
   "integers (the list always has at least one element).",
   {"function_name": "max_subarray_sum", "tests": [
       {"args": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6},
       {"args": [[1, 2, 3, 4]], "expected": 10},
       {"args": [[-1, -2, -3]], "expected": -1},
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
