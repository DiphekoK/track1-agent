"""
Executes a generated/fixed function against real test cases in a
subprocess, instead of asking a judge model to eyeball whether code
"looks right". Used by data/label_dataset.py to grade code_debug and
code_gen answers.

Dev-time tooling only - runs against code the local model wrote on
your own machine while building the training set. Not part of the
submitted container, never runs on untrusted input.
"""
import re
import subprocess
import sys
import tempfile

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(answer_text):
    match = CODE_BLOCK_RE.search(answer_text)
    return match.group(1) if match else answer_text


def run_tests(answer_text, function_name, tests, timeout=10):
    code = extract_code(answer_text)
    harness = (
        code + "\n\n"
        f"_tests = {tests!r}\n"
        f"_fn = {function_name}\n"
        "_all_ok = True\n"
        "for _t in _tests:\n"
        "    try:\n"
        "        if _fn(*_t['args']) != _t['expected']:\n"
        "            _all_ok = False\n"
        "    except Exception:\n"
        "        _all_ok = False\n"
        "print('PASS' if _all_ok else 'FAIL')\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness)
        path = f.name

    try:
        result = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return result.stdout.strip().endswith("PASS")
