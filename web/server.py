"""
Local backend for the web/ demo UI. Reuses the real project modules
(categories, local_llm, fireworks_client, baseline_router,
router/infer_router) directly, so the demo shows genuinely real routing
decisions and real answers - not a browser-side simulation. This is for
recording a presentation against the actual system, not a public
deployment (no auth, binds to localhost only).

Run from the repo root so the relative model paths in .env still resolve:
    python web/server.py
Then open http://localhost:8787/ in a browser.
"""
import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_dotenv():
    # same convenience .env loading data/label_dataset.py's neighbors expect
    # you to have already exported - just done here too so `python
    # web/server.py` works without a separate export step
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

import categories
import local_llm
import fireworks_client
import baseline_router
import agent
from router.infer_router import available as router_available, predict as router_predict

# Most host platforms (Render, Railway, Fly.io...) assign the port via
# $PORT and expect the app to bind 0.0.0.0, not localhost - DEMO_PORT is
# kept as a fallback name for local runs that set it explicitly.
PORT = int(os.environ.get("PORT") or os.environ.get("DEMO_PORT", "8787"))
HOST = os.environ.get("DEMO_HOST", "0.0.0.0")
WEB_DIR = Path(__file__).resolve().parent


def estimate_tokens(s):
    return max(1, round(len(s or "") / 3.6))


def get_config():
    local_label = Path(local_llm.MODEL_PATH).stem
    try:
        fireworks_label = agent.get_fireworks_model().rsplit("/", 1)[-1]
        fireworks_error = None
    except KeyError as e:
        fireworks_label = None
        fireworks_error = f"missing env var {e}"
    return {
        "localModel": local_label,
        "fireworksModel": fireworks_label,
        "fireworksError": fireworks_error,
        "routerAvailable": router_available(),
        "routerConfidenceThreshold": agent.ROUTER_CONFIDENCE_THRESHOLD,
    }


def run_pipeline(prompt):
    category = categories.classify(prompt)

    # Step 1: the real finetuned router (or heuristic fallback), zero cost -
    # the same decision agent.decide_backend makes, just surfaced here
    # instead of staying an internal choice
    confidence = None
    note = None
    if router_available():
        label, confidence = router_predict(prompt)
        if confidence < agent.ROUTER_CONFIDENCE_THRESHOLD:
            note = f"router unsure ({confidence:.2f}), used heuristic instead"
            label = "local" if categories.should_use_local(category) else "fireworks"
    else:
        note = "trained router weights not available, used heuristic"
        label = "local" if categories.should_use_local(category) else "fireworks"

    finetuned = {"backend": label, "confidence": confidence, "note": note}

    # Step 2: the real baseline - an actual Fireworks call just to classify,
    # same as baseline_router.py, costs real tokens to decide
    try:
        baseline_label, baseline_tokens = baseline_router.classify(prompt)
        baseline = {"backend": baseline_label, "tokens": baseline_tokens, "error": None}
    except Exception as e:
        baseline = {"backend": None, "tokens": 0, "error": str(e)}

    # Step 3: answer via whichever backend the finetuned router chose in step 1
    try:
        if label == "local":
            text, tokens = local_llm.answer_with_usage(prompt, category)
            if tokens is None:
                tokens = estimate_tokens(prompt) + estimate_tokens(text)
        else:
            system = fireworks_client.SYSTEM_PROMPTS.get(category)
            resp = fireworks_client.chat(agent.get_fireworks_model(), prompt, system=system, max_tokens=400)
            text, tokens = resp["text"], resp["total_tokens"]
        answer = {"backend": label, "text": text, "tokens": tokens, "error": None}
    except Exception as e:
        traceback.print_exc()
        answer = {"backend": label, "text": None, "tokens": 0, "error": str(e)}

    return {"category": category, "finetuned": finetuned, "baseline": baseline, "answer": answer}


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/config":
            self._send_json(get_config())
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path != "/pipeline":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid JSON body"}, status=400)
            return
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            self._send_json({"error": "empty prompt"}, status=400)
            return
        try:
            self._send_json(run_pipeline(prompt))
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=500)

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} - {fmt % args}")


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    cfg = get_config()
    print(f"Query Router demo backend on http://{HOST}:{PORT}  (Ctrl+C to stop)")
    print(f"  local model:     {cfg['localModel']}")
    print(f"  fireworks model: {cfg['fireworksModel'] or '(not configured: ' + str(cfg['fireworksError']) + ')'}")
    print(f"  trained router:  {'available' if cfg['routerAvailable'] else 'not available (heuristic fallback)'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
