#!/usr/bin/env python3
"""Telemetry + console server on :18400.

The runner owns the data and writes state.json atomically (os.replace); this process
only reads it and serves it. Keeping the writer out of the web process means a page
bug can at worst produce wrong pixels, never a wrong measurement.

The console adds ONE write: POST /api/dispatch drops a job file into a queue. This
process never executes a job, never switches a model and never spawns a runner --
that is a separate systemd unit (load-dispatch.service) under an exclusive flock.
Both write routes require a bearer token from a root-only file; every read route is
open, because the dashboard is browsed from a phone.

`now_age_s` is computed here from the file mtime so the page can show an honest
staleness badge instead of pretending the run is live when it is not.
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corrected_metrics import overlay as _correct   # noqa: E402
import console_core as C                            # noqa: E402
import live_metrics as LM                           # noqa: E402

LOAD = Path(os.environ.get("RUNBOARD_LOAD", "/root/load"))
# Static pages live beside the modules in the deployment and under static/ in a checkout.
STATIC = Path(os.environ.get("RUNBOARD_STATIC", str(LOAD)))
RUN = LOAD / "run"
HTML = STATIC / "index.html"
HISTORY_HTML = STATIC / "history.html"
CONSOLE_HTML = STATIC / "console.html"
PORT = 18400
MAX_BODY = 64 * 1024
AUTH_LOG = LOAD / "dispatch/auth.log"


def auth_log(line: str) -> None:
    try:
        AUTH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with AUTH_LOG.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                     f"{line}\n")
    except Exception:
        pass


class Handler(BaseHTTPRequestHandler):
    server_version = "zgx-console/1"

    # ---------------------------------------------------------------- writes

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/dispatch", "/api/download"):
            self._send(404, "application/json", b'{"error":"no such route"}')
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except Exception:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._send(400, "application/json",
                       json.dumps({"error": f"body must be 1..{MAX_BODY} bytes"}).encode())
            return
        raw = self.rfile.read(length)
        addr = self.client_address[0] if self.client_address else "?"

        # Fail closed: no token configured means no writes, not open writes.
        if not C.console_token():
            auth_log(f"REFUSED {path} from {addr}: no token configured on this box")
            self._send(503, "application/json",
                       b'{"error":"no dispatch token configured on the server"}')
            return
        if not C.token_ok(self.headers.get("Authorization")):
            auth_log(f"DENIED  {path} from {addr}: bad or missing bearer token")
            self._send(401, "application/json", b'{"error":"unauthorized"}')
            return

        try:
            job = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send(400, "application/json",
                       json.dumps({"error": f"bad JSON: {exc}"}).encode())
            return

        if path == "/api/dispatch":
            job.setdefault("action", "eval")
            job.setdefault("engine", "current")
            job.setdefault("job_id", C.new_job_id("eval", str(job.get("preset", ""))))
        else:
            job["action"] = "download"
            job.setdefault("job_id", C.new_job_id("dl", str(job.get("repo", "")).replace("/", "_")))
        job.setdefault("created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        job["_queued_from"] = addr

        ok, why = C.validate_job(job)
        if not ok:
            auth_log(f"REJECT  {path} from {addr}: {why}")
            self._send(422, "application/json",
                       json.dumps({"error": why, "job": job}).encode())
            return

        C.QUEUE.mkdir(parents=True, exist_ok=True)
        C.atomic_json(C.QUEUE / f"{job['job_id']}.json", job)
        auth_log(f"QUEUED  {job['job_id']} {job['action']} from {addr}")
        self._send(202, "application/json",
                   json.dumps({"queued": True, "job_id": job["job_id"],
                               "job": job,
                               "poll": "/api/dispatch"}).encode())

    # ---------------------------------------------------------------- reads

    def do_GET(self):
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)

        if path.startswith("/api/state"):
            p = RUN / "state.json"
            try:
                st = json.loads(p.read_text())
                # Serve the CORRECTED throughput view. The runner recorded an
                # "aggregate" that divided by sum(per-request elapsed) -- the
                # per-stream mean under concurrency -- so it fell as concurrency
                # rose. Correcting here keeps the page honest without touching
                # recorded data.
                try:
                    st = _correct(st)
                except Exception as exc:
                    st["correction"] = {"applied": False, "error": type(exc).__name__}
                st["now_age_s"] = round(time.time() - p.stat().st_mtime, 1)
                self._json(st)
            except Exception as exc:
                self._send(503, "application/json",
                           json.dumps({"error": type(exc).__name__,
                                       "detail": "no state yet"}).encode())

        elif path.startswith("/api/history"):
            # served straight from the collector's file: this process never builds
            # the index, so a bug here cannot corrupt history
            try:
                self._send(200, "application/json", (RUN / "history.json").read_bytes())
            except Exception as exc:
                self._send(503, "application/json",
                           json.dumps({"error": type(exc).__name__,
                                       "detail": "history not built yet"}).encode())

        elif path.startswith("/api/models"):
            try:
                self._send(200, "application/json", C.MODELS.read_bytes())
            except Exception:
                self._send(503, "application/json",
                           b'{"error":"models.json not built yet - run registry.py"}')

        elif path.startswith("/api/live"):
            # Read-only, no token: this is the "grafana panel" surface. Values come from
            # the local exporter, history from Prometheus; either can be down and the
            # document says so rather than filling the gap with plausible numbers.
            try:
                self._json(LM.live_doc(window=(q.get("window") or [None])[0]))
            except Exception as exc:
                self._send(502, "application/json",
                           json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode())

        elif path.startswith("/api/dispatch"):
            out = {"status": C.read_json(C.STATUS, {"state": "unknown"}),
                   "history": C.read_json(C.HISTORY, {"jobs": []}),
                   "queue": sorted(p.name for p in C.QUEUE.glob("*.json"))}
            self._json(out)

        elif path.startswith("/api/hf/search"):
            query = (q.get("q") or [""])[0].strip()
            if not query:
                self._send(400, "application/json", b'{"error":"q is required"}')
                return
            try:
                limit = max(1, min(int((q.get("limit") or ["12"])[0]), 40))
            except Exception:
                limit = 12
            try:
                self._json({"query": query, "results": C.hf_search(query, limit)})
            except Exception as exc:
                self._send(502, "application/json",
                           json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode())

        elif path.startswith("/api/hf/plan"):
            repo = (q.get("repo") or [""])[0].strip()
            if "/" not in repo:
                self._send(400, "application/json",
                           b'{"error":"repo must look like owner/name"}')
                return
            try:
                self._json(C.hf_plan(repo))
            except Exception as exc:
                self._send(502, "application/json",
                           json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode())

        elif path in ("/console", "/console.html"):
            self._page(CONSOLE_HTML)
        elif path in ("/history", "/history.html"):
            self._page(HISTORY_HTML)
        elif path in ("/", "/index.html"):
            self._page(HTML)
        elif path == "/health":
            self._send(200, "application/json", b'{"ok":true}')
        else:
            self._send(404, "text/plain", b"not found")

    # ---------------------------------------------------------------- plumbing

    def _page(self, path: Path):
        try:
            self._send(200, "text/html; charset=utf-8", path.read_bytes())
        except Exception:
            self._send(500, "text/plain", f"{path.name} missing".encode())

    def _json(self, obj):
        self._send(200, "application/json", json.dumps(obj, default=str).encode())

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
