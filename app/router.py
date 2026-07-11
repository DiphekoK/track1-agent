"""
Figures out which capability category a prompt belongs to, and whether
we can answer it with the local model or need to spend Fireworks tokens.

This is just keyword/regex based - no ML here. Good enough for routing,
doesn't need to be perfect since the categories are pretty distinct in
practice. If it misclassifies something occasionally that's fine, it'll
just get handled by the "wrong" backend and probably still be okay.
"""
import re

# Categories that a small local model can handle reasonably well.
# These lean toward "retrieve/transform" tasks rather than multi-step reasoning.
LOCAL_CATEGORIES = {"factual", "sentiment", "summarization", "ner"}

# Categories that go to Fireworks because small local models tend to
# fall over on multi-step reasoning / code correctness.
FIREWORKS_CATEGORIES = {"math", "logic", "code_debug", "code_gen"}

_CODE_DEBUG_RE = re.compile(r"\b(bug|debug|fix (it|the bug|this)|broken|doesn't work|incorrect (output|result))\b", re.I)
_CODE_GEN_RE = re.compile(r"\bwrite (a|an|the)\b.{0,40}\b(function|method|program|script|class)\b", re.I)
_LOGIC_RE = re.compile(r"\b(each own|who owns|each has a different|puzzle|exactly one of|must be true|which one of them)\b", re.I)
_MATH_RE = re.compile(r"(\d+\s*%|\bpercent(age)?\b|\bhow many\b|\bhow much\b|\btotal\b|\bremain(ing)?\b|\baverage\b|\bsum of\b|\bprofit\b|\bdiscount\b)", re.I)
_NER_RE = re.compile(r"\b(named entit(y|ies)|extract (all )?(the )?entit(y|ies)|identify (the )?(people|organizations|locations|dates))\b", re.I)
_SENTIMENT_RE = re.compile(r"\bsentiment\b", re.I)
_SUMMARY_RE = re.compile(r"\b(summari[sz]e|summari[sz]ation|condense|tl;?dr)\b", re.I)

# has to have actual code-ish content for it to count as a debug task,
# otherwise "this doesn't work" style phrasing in a non-code prompt would
# get misrouted
_CODE_SNIPPET_RE = re.compile(r"(def |function\s*\w*\s*\(|=>|\{|\breturn\b|console\.log|print\()")


def classify(prompt: str) -> str:
    p = prompt.strip()

    if _CODE_DEBUG_RE.search(p) and _CODE_SNIPPET_RE.search(p):
        return "code_debug"

    if _CODE_GEN_RE.search(p):
        return "code_gen"

    if _LOGIC_RE.search(p):
        return "logic"

    if _NER_RE.search(p):
        return "ner"

    if _SENTIMENT_RE.search(p):
        return "sentiment"

    if _SUMMARY_RE.search(p):
        return "summarization"

    if _MATH_RE.search(p):
        return "math"

    return "factual"


def should_use_local(category: str) -> bool:
    return category in LOCAL_CATEGORIES
