"""A deterministic stand-in for the OpenAI-compatible chat endpoint.

Serves ``GET /v1/models`` and ``POST /v1/chat/completions`` on localhost and answers the
judge prompts of the evaluation (FS, GS, Coherence, the MINE judge) and the DSPy
deduplication prompts of KGGen from a hash of the judged item, so every run gives the same
labels without network access. The server counts requests, which lets tests check that a
resumed run sends none.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "stub-judge"


def _bucket(text: str, n: int) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % n


def _last_field(prompt: str, label: str) -> str:
    """Text after the last ``label`` line of a prompt (the item being judged)."""
    idx = prompt.rfind(label)
    if idx < 0:
        return ""
    return prompt[idx + len(label):].split("\n", 1)[0].strip()


def fs_label(triple_json: str) -> bool:
    return _bucket("fs" + triple_json, 3) != 0


def coherence_label(triple_json: str) -> bool:
    return _bucket("coh" + triple_json, 4) != 0


def gs_split(triple_json: str) -> int:
    """Number of sub-triples the stub judge proposes: 0, 2 or 3."""
    return (0, 2, 3)[_bucket("gs" + triple_json, 3)]


def answer(prompt: str) -> str:
    prompt = prompt.replace("\n/no_think", "")
    if "[[ ## item ## ]]" in prompt:  # KGGen LLM deduplication (DSPy chat adapter)
        item = prompt.rsplit("[[ ## item ## ]]", 1)[1].split("[[ ##", 1)[0].strip()
        return (f"[[ ## duplicates ## ]]\n[]\n\n[[ ## alias ## ]]\n{item}\n\n"
                "[[ ## completed ## ]]")
    tail = prompt.rstrip()
    if tail.endswith("Factualness:"):
        triple = _last_field(prompt, "Relationship:")
        # vary the surface form so the response parser is exercised
        return "true" if fs_label(triple) else "The answer is false."
    if tail.endswith("Coherence:"):
        triple = _last_field(prompt, "Relationship:")
        return "coherent" if coherence_label(triple) else "Incoherent."
    if tail.endswith("Sub-triples:"):
        triple = _last_field(prompt, "Triple:")
        n = gs_split(triple)
        if n == 0:
            return "N/A (already atomic)\nGranularity: 0"
        subs = "\n".join(f'["part {i}", "r", "o"]' for i in range(n))
        return f"{subs}\nGranularity: {n}"
    if tail.endswith("Answer (1 or 0):"):
        context = prompt.rsplit("CONTEXT:", 1)[1].split("FACT:", 1)[0].lower()
        fact = prompt.rsplit("FACT:", 1)[1].split("Answer (1 or 0):", 1)[0].lower()
        words = [w for w in re.findall(r"[a-z]{5,}", fact)]
        hit = sum(w in context for w in words)
        return "1" if words and hit >= max(1, len(words) // 3) else "0"
    return "unknown"


class StubLLMServer:
    """Context manager running the stub endpoint in a background thread."""

    def __init__(self, fail_first: int = 0):
        self.requests = 0
        self._fail_left = fail_first
        self._lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _send(self, code: int, payload: dict):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"object": "list", "data": [
                        {"id": MODEL_ID, "object": "model", "root": MODEL_ID}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
                with server._lock:
                    server.requests += 1
                    fail = server._fail_left > 0
                    if fail:
                        server._fail_left -= 1
                if fail:
                    self._send(500, {"error": {"message": "injected failure"}})
                    return
                content = req["messages"][-1]["content"]
                if isinstance(content, list):  # DSPy may send content parts
                    content = "".join(p.get("text", "") for p in content)
                self._send(200, {
                    "id": "cmpl-stub", "object": "chat.completion", "created": 0,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": answer(content)}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                })

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1/"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "StubLLMServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


if __name__ == "__main__":  # run standalone: python tests/llm_stub.py
    import time
    with StubLLMServer() as srv:
        print(srv.url, flush=True)
        while True:
            time.sleep(3600)
