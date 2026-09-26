#!/usr/bin/env python3
"""zgx-exporter — Prometheus exporter for the DGX Spark (GB10) inference box.

Why this exists: Prometheus pulls from a scrape target, and the EXL3 serving path
(exllamav3's stdlib http.server) has no /metrics endpoint at all, so nothing on
.107 can be scraped while EXL3 is serving. The previous vLLM target
(http://192.168.1.107:18300/metrics) is therefore down for the whole EXL3 window,
which is why the ZGX dashboard had no data.

This serves, on one port:
  * live machine metrics, sampled at scrape time (GPU power/temperature/clock,
    utilisation, memory, swap) — so the box is observable whether or not a load
    run is in progress;
  * load-run metrics, read from the load soak's state.json when it exists;
  * a serving health gauge for :18300.

Stdlib only, deliberately: it must run under the host python with no venv, and a
web bug here can only ever produce wrong numbers, never touch the run.

Honesty rules followed here:
  * a metric that cannot be sampled is OMITTED, never reported as 0 — a fake zero
    on a power or throughput graph is indistinguishable from a real idle reading;
  * `zgx_load_state_age_seconds` is exposed so a stale run is visible as stale.
"""
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corrected_metrics import overlay as _correct   # noqa: E402

PORT = 9400
STATE = Path(os.environ.get("RUNBOARD_LOAD", "/root/load")) / "run/state.json"
DISPATCH = Path(os.environ.get("RUNBOARD_LOAD", "/root/load")) / "dispatch/status.json"
SERVE_HEALTH = "http://127.0.0.1:18300/v1/models"
GPU_QUERY = ("power.draw,temperature.gpu,utilization.gpu,clocks.current.graphics,"
             "clocks_throttle_reasons.active")


def gauge(name, value, labels=None, help_text=""):
    if value is None:
        return []                              # omit, never fake a zero
    lbl = ""
    if labels:
        inner = ",".join(f'{k}="{v}"' for k, v in labels.items())
        lbl = "{" + inner + "}"
    out = []
    if help_text:
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} gauge")
    out.append(f"{name}{lbl} {value}")
    return out


def sample_gpu():
    """One nvidia-smi call per scrape. Returns {} on failure (metrics omitted)."""
    try:
        raw = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8).stdout.strip()
        if not raw:
            return {}
        parts = [p.strip() for p in raw.split(",")]
        out = {}
        try:
            out["power"] = float(parts[0])
            out["temp"] = float(parts[1])
            out["util"] = float(parts[2])
        except ValueError:
            return {}
        out["clock"] = parts[3]
        out["throttle"] = parts[4]
        return out
    except Exception:
        return {}


def sample_mem():
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            kb = int(v.strip().split()[0])
            if k == "MemAvailable":
                out["mem_available"] = kb * 1024
            elif k == "SwapFree":
                out["swap_free"] = kb * 1024
            elif k == "SwapTotal":
                out["swap_total"] = kb * 1024
    except Exception:
        pass
    return out


def serving_up():
    import urllib.request
    try:
        with urllib.request.urlopen(SERVE_HEALTH, timeout=4) as r:
            return 1 if r.status == 200 else 0
    except Exception:
        return 0


def load_metrics():
    """Load-run metrics, or [] when no run state exists yet."""
    try:
        st = json.loads(STATE.read_text())
    except Exception:
        return []
    # Recompute throughput from the raw request log before publishing. The runner's
    # recorded "aggregate" divided by sum(per-request elapsed) — the per-stream mean
    # under concurrency — so it FELL as throughput-per-wall-clock held flat, and its
    # scaling figure was pinned near 1.0 at every concurrency (fixed 2026-09-24).
    try:
        st = _correct(st)
    except Exception:
        pass
    lines = []
    age = time.time() - STATE.stat().st_mtime
    lines += gauge("zgx_load_state_age_seconds", round(age, 1),
                   help_text="seconds since the load run last updated its state")
    lines += gauge("zgx_load_live_aggregate_tok_s", st.get("live_agg_tps"),
                   help_text="aggregate decode tok/s = sum(completion tokens finishing in "
                             "the trailing window) / window seconds, recomputed from the "
                             "raw request log")
    lines += gauge("zgx_load_live_per_stream_tok_s", st.get("live_stream_tps"),
                   help_text="mean per-stream decode tok/s over the trailing window")
    t = st.get("totals") or {}
    lines += gauge("zgx_load_requests_total", t.get("requests"))
    lines += gauge("zgx_load_errors_total", t.get("errors"))
    lines += gauge("zgx_load_completion_tokens_total", t.get("completion_tokens"))
    lines += gauge("zgx_load_elapsed_seconds_total",
                   round(t["elapsed"], 2) if t.get("elapsed") else None)

    idx = st.get("phase_index")
    phases = st.get("phases") or []
    if idx is not None and 0 <= idx < len(phases):
        lines += gauge("zgx_load_phase_concurrency", phases[idx].get("c"),
                       help_text="concurrency level of the phase currently running")
        lines += gauge("zgx_load_phase_minutes", phases[idx].get("minutes"))
    running = 0 if st.get("phase") == "DONE" else 1
    lines += gauge("zgx_load_running", running,
                   help_text="1 while the load run is in a phase, 0 once finished")

    # per-phase results become a labelled series so Grafana can table them
    for ph in phases:
        r = ph.get("result")
        if not r:
            continue
        labels = {"phase": str(ph.get("name")), "concurrency": str(ph.get("c"))}
        for key, metric in (("aggregate_tok_s", "zgx_load_phase_aggregate_tok_s"),
                            ("per_stream_mean", "zgx_load_phase_per_stream_mean_tok_s"),
                            ("per_stream_p50", "zgx_load_phase_per_stream_p50_tok_s")):
            lines += gauge(metric, r.get(key), labels)
        lines += gauge("zgx_load_phase_requests", r.get("requests"), labels)
        lines += gauge("zgx_load_phase_errors", r.get("errors"), labels)
        # Prefer the shared definition (aggregate / per-stream MEAN) from
        # corrected_metrics so the dashboard and the live page cannot disagree about
        # what "scaling" means. Fall back to p50 only if it is absent.
        scale = r.get("scaling_x")
        if scale is None:
            agg, p50 = r.get("aggregate_tok_s"), r.get("per_stream_p50")
            scale = round(agg / p50, 3) if (agg and p50) else None
        lines += gauge("zgx_load_phase_scaling_x", scale, labels,
                       help_text="aggregate wall-clock tok/s divided by per-stream mean tok/s")
    return lines


def dispatch_metrics():
    """What the dispatcher itself is doing, from its own status file.

    `zgx_load_running` answers "is the SOAK running" -- it is written by the soak
    harness, so during an eval run it reads 0 while the box sits at 96% GPU and
    the state-age gauge grows by the hour. These gauges answer the question an
    operator actually means: is a job executing on this box right now?
    """
    try:
        st = json.loads(DISPATCH.read_text())
    except Exception:
        return []
    state = st.get("state") or "unknown"
    job = st.get("job") or {}
    lines = []
    lines += gauge("zgx_dispatch_active", 1 if state == "running" else 0,
                   help_text="1 while the dispatcher is executing a job (eval or soak)")
    try:
        age = round(time.time() - DISPATCH.stat().st_mtime, 1)
    except Exception:
        age = None
    lines += gauge("zgx_dispatch_state_age_seconds", age,
                   help_text="seconds since the dispatcher last wrote its status")
    lines += gauge("zgx_dispatch_queue_depth", st.get("queue_depth"),
                   help_text="queued job files not yet picked up")
    labels = {"state": state}
    if job.get("preset"):
        labels["preset"] = str(job["preset"])
    lines += gauge("zgx_dispatch_state", 1, labels=labels,
                   help_text="current dispatcher state as a labelled series")
    return lines


def collect():
    lines = []
    gpu = sample_gpu()
    lines += gauge("zgx_gpu_power_watts", gpu.get("power"), help_text="GPU board power draw")
    lines += gauge("zgx_gpu_temperature_celsius", gpu.get("temp"), help_text="GPU temperature")
    lines += gauge("zgx_gpu_utilization_percent", gpu.get("util"), help_text="GPU utilisation")
    if gpu.get("clock"):
        lines += gauge("zgx_gpu_clock_mhz", float(gpu["clock"]), help_text="GPU graphics clock")
    mem = sample_mem()
    lines += gauge("zgx_memory_available_bytes", mem.get("mem_available"))
    lines += gauge("zgx_swap_free_bytes", mem.get("swap_free"))
    lines += gauge("zgx_swap_total_bytes", mem.get("swap_total"))
    lines += gauge("zgx_serving_up", serving_up(),
                   help_text="1 if the inference endpoint on :18300 answers health")
    lines += load_metrics()
    lines += dispatch_metrics()
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/metrics"):
            body = collect().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass
        elif self.path == "/health":
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
