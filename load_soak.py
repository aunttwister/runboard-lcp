#!/usr/bin/env python3
"""load-soak — sustained mixed-shape load against the deployed EXL3 2.50bpw service.

Why this shape: the question is not "is it accurate" but "how does it behave when
it is busy". So the workload is a mix of the request shapes this box actually
serves (short chat, code generation, long-document summarisation, tool-call JSON,
deep reasoning), run at rising concurrency and then held at a steady load.

Metric definitions follow docs/PROCEDURES.md §4 exactly:
  e2e_tok_s = completion_tokens / elapsed_seconds, per row, reported as
  mean / p50 / aggregate (sum completion tokens / sum elapsed). Never derived
  from content length, never from a possibly-empty wrapper field.

Telemetry is sampled on the serve host at a 2 s cadence — power, temperature,
utilisation, clocks, throttle reasons, /proc/meminfo — and summarised load-only
(util > 0). On GB10 utilisation is effectively binary, so power is the real load
discriminator; both are recorded.

Honest coverage gap: the server refuses `stream: true` ("streaming not
implemented"), so time-to-first-token cannot be measured from the client. Only
end-to-end rate is claimed.
"""
import collections
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = os.environ.get("LOAD_BASE", "http://127.0.0.1:18300")
MODEL = os.environ.get("LOAD_MODEL", "qwen38-flash-next-exl3")
KIT = Path("/root/exl3-engine/r0b0bench/subsets/q200v2/artifacts/quality-text-180-v2.jsonl")
OUT = Path(os.environ.get("LOAD_OUT", "/root/load/run"))
KW = {"enable_thinking": True, "reasoning_effort": "low", "thinking": True}


def _phases():
    """Phases are overridable so the harness can be smoke-tested cheaply against a
    different endpoint (proving the client path) without loading the real service."""
    raw = os.environ.get("LOAD_PHASES")
    if raw:
        return [tuple(x) for x in json.loads(raw)]
    return [
        ("c1-baseline", 1, 10),
        ("c2", 2, 12),
        ("c4", 4, 15),
        ("c8", 8, 15),
        ("soak-c4", 4, 45),
    ]


PHASES = _phases()
SHAPES = ["chat_short", "code_gen", "doc_summary", "tool_call", "deep_reason"]
WALL_CAP_MIN = 130

_state_lock = threading.Lock()
_state = {
    "started_utc": None, "now_utc": None, "phase": None, "phase_index": 0,
    "phases": [{"name": n, "c": c, "minutes": m, "done": False} for n, c, m in PHASES],
    "series": {"ts": [], "agg_tps": [], "stream_tps": [], "gpu_power": [],
               "gpu_temp": [], "gpu_util": [], "mem_avail_gb": [], "inflight": []},
    "totals": {"requests": 0, "errors": 0, "completion_tokens": 0, "elapsed": 0.0},
    "live_agg_tps": None, "live_stream_tps": None, "live_window_s": 60,
    "shapes": {},
}
SERIES_MAX = 900
# rolling window of recently completed requests, for a live (not per-phase) rate
_recent = collections.deque(maxlen=800)


def _push(key, value):
    s = _state["series"][key]
    s.append(value)
    if len(s) > SERIES_MAX:
        del s[:-SERIES_MAX]


def write_state():
    tmp = OUT / "state.json.tmp"
    tmp.write_text(json.dumps(_state))
    os.replace(tmp, OUT / "state.json")


def load_prompts():
    rows = []
    for line in KIT.read_text(errors="replace").splitlines():
        line = line.strip().rstrip(",")
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    by = {}
    for r in rows:
        by.setdefault(r.get("family"), []).append(r)
    return by


def build_shapes(by):
    gsm = [r["prompt"] for r in by.get("gsm8k", []) if r.get("prompt")]
    hard = [r["prompt"] for r in by.get("hard_reasoning", []) if r.get("prompt")]
    code = [r["prompt"] for r in by.get("humaneval", []) if r.get("prompt")]
    ife = [r["prompt"] for r in by.get("ifeval", []) if r.get("prompt")]
    # a long prompt built from frozen kit text: same tokenizer path as the kit
    blob = "\n\n".join(ife + gsm)[:32000]
    longdoc = ("Summarise the following material in about 300 words, then list the "
               "three most important items.\n\n" + blob)
    return {
        "chat_short": (gsm or [""], 256),
        "deep_reason": (hard or [""], 2048),
        "code_gen": (code or [""], 1536),
        "tool_call": ([p + "\n\nReply with a single JSON object only." for p in (ife or [""])], 512),
        "doc_summary": ([longdoc], 700),
    }


def post(prompt, max_tokens, timeout=900):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": KW,
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
        elapsed = time.time() - t0
        usage = payload.get("usage") or {}
        ct = int(usage.get("completion_tokens") or 0)
        pt = int(usage.get("prompt_tokens") or 0)
        fr = (payload.get("choices") or [{}])[0].get("finish_reason")
        return {"ok": True, "prompt_tokens": pt, "completion_tokens": ct,
                "elapsed": elapsed, "finish_reason": fr,
                "e2e_tok_s": (ct / elapsed) if elapsed > 0 else None}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "elapsed": time.time() - t0}


def telemetry(stop, log):
    """2 s cadence on the serve host. Load-only gating happens at summary time."""
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,utilization.gpu,"
                 "clocks.current.graphics,clocks_throttle_reasons.active",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8).stdout.strip()
            parts = [p.strip() for p in out.split(",")]
            power, temp, util = float(parts[0]), float(parts[1]), float(parts[2])
            clock = parts[3]
            throttle = parts[4]
            mem = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1)
                if k in ("MemAvailable", "SwapFree", "SwapTotal"):
                    mem[k] = int(v.strip().split()[0]) / 1048576.0
            with log.open("a") as fh:
                fh.write(json.dumps({"ts": time.time(), "power_w": power, "temp_c": temp,
                                     "util_pct": util, "clock_mhz": clock,
                                     "throttle": throttle,
                                     "mem_avail_gb": mem.get("MemAvailable"),
                                     "swap_free_gb": mem.get("SwapFree")}) + "\n")
            with _state_lock:
                _push("gpu_power", power)
                _push("gpu_temp", temp)
                _push("gpu_util", util)
                _push("mem_avail_gb", mem.get("MemAvailable"))
                # a live rolling rate, so the page keeps moving inside a long phase
                now = time.time()
                win = float(_state.get("live_window_s") or 60)
                recent = [r for r in _recent if now - r[0] <= win]
                if recent:
                    # aggregate = tokens completed INSIDE the window / the window's own
                    # duration. Dividing by sum(elapsed) (the original bug) yielded the
                    # per-stream mean instead — 24.05 tok/s shown at c=4 on 2026-09-24
                    # where the box was actually emitting ~96.
                    _state["live_agg_tps"] = round(sum(r[1] for r in recent) / win, 2)
                    _state["live_stream_tps"] = round(
                        statistics.mean([r[1] / r[2] for r in recent if r[2] > 0]), 2)
                else:
                    _state["live_agg_tps"] = None
                    _state["live_stream_tps"] = None
                _push("agg_tps", _state["live_agg_tps"])
                _push("stream_tps", _state["live_stream_tps"])
                _push("ts", now)
                write_state()
        except Exception:
            pass
        time.sleep(2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by = load_prompts()
    shapes = build_shapes(by)
    reqs = (OUT / "requests.jsonl").open("w")
    errlog = (OUT / "errors.jsonl").open("w")
    tlog = OUT / "telemetry.jsonl"
    if tlog.exists():
        tlog.unlink()
    stop_tel = threading.Event()
    tel = threading.Thread(target=telemetry, args=(stop_tel, tlog), daemon=True)
    tel.start()

    started = time.time()
    _state["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    write_state()   # publish immediately so the live page has something to show
    phase_results = []

    for idx, (name, conc, minutes) in enumerate(PHASES):
        _state["phase"] = name
        _state["phase_index"] = idx
        deadline = time.time() + minutes * 60
        t_start = time.time()
        window = []          # (ts, completion_tokens, elapsed)
        errs = 0
        worker_errors = []
        print(f"[load] phase {name}: concurrency={conc} for {minutes} min", flush=True)

        shapes_min = shapes

        def worker():
            """One load generator. Any exception here would otherwise vanish into a
            future that nobody reads — which is exactly how a run reports 0 requests
            while having actually served traffic. So: catch, record, and fail loudly."""
            nonlocal errs
            i = 0
            while time.time() < deadline:
                if (time.time() - started) > WALL_CAP_MIN * 60:
                    return
                shape = SHAPES[i % len(SHAPES)]
                pool, mt = shapes_min[shape]
                prompt = pool[i % len(pool)]
                res = post(prompt, mt)
                i += 1
                rec = {"ts": time.time(), "phase": name, "concurrency": conc,
                       "shape": shape, "max_tokens": mt, **res}
                if res["ok"]:
                    reqs.write(json.dumps(rec) + "\n")
                    reqs.flush()
                elif errs < 200:
                    # log the first 200 failures, count the rest: a dead endpoint
                    # otherwise spins ~28k/s and floods this file with hundreds of MB
                    errlog.write(json.dumps(rec) + "\n")
                    errlog.flush()
                if not res["ok"]:
                    with _state_lock:
                        _state["totals"]["errors"] += 1
                    errs += 1
                    time.sleep(0.5)        # never hot-spin against a broken endpoint
                    continue
                ts = rec["ts"]                 # completion timestamp, not a key on `res`
                ct = res["completion_tokens"]
                el = res["elapsed"]
                with _state_lock:
                    window.append((ts, ct, el))
                    _recent.append((ts, ct, el))
                    _state["totals"]["requests"] += 1
                    _state["totals"]["completion_tokens"] += ct
                    _state["totals"]["elapsed"] += el
                    sh = _state["shapes"].setdefault(shape, {"n": 0, "tokens": 0, "elapsed": 0.0})
                    sh["n"] += 1
                    sh["tokens"] += ct
                    sh["elapsed"] += el

        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [ex.submit(worker) for _ in range(conc)]
            for f in futs:
                try:
                    f.result()
                except Exception as exc:
                    worker_errors.append(f"{type(exc).__name__}: {exc}")

        wall = time.time() - t_start
        # TRUE aggregate throughput = total tokens produced / WALL-CLOCK duration.
        #
        # This originally divided by the SUM of per-request elapsed times:
        #     sum(c)/sum(e)
        # For `conc` concurrent requests of similar length that equals `c/e` — the
        # per-stream mean, not an aggregate. Two consequences, both observed live on
        # 2026-09-24: it read 24.05 tok/s at c=4 where the box was really producing
        # ~96, and `scaling_x = agg / per_stream_mean` was algebraically pinned to
        # ~1.00 at EVERY concurrency, so the scaling column could never show scaling.
        # Dividing by wall time is the only definition that answers "how much text
        # does the box emit per second", which is the question the run exists to ask.
        total_tokens = sum(c for _, c, _ in window)
        agg = total_tokens / wall if wall > 0 else 0.0
        per_stream = [c / e for _, c, e in window if e > 0]
        busy = sum(e for _, _, e in window)
        _state["phases"][idx]["done"] = True
        res = {"phase": name, "concurrency": conc, "minutes": minutes,
               "requests": len(window), "errors": errs,
               "wall_seconds": round(wall, 2),
               "total_completion_tokens": total_tokens,
               "aggregate_tok_s": round(agg, 2),
               "per_stream_mean": round(statistics.mean(per_stream), 2) if per_stream else None,
               "per_stream_p50": round(statistics.median(per_stream), 2) if per_stream else None,
               "scaling_x": round(agg / (statistics.mean(per_stream) or 1), 2) if per_stream else None,
               "stream_utilization": round(busy / (wall * conc), 3) if wall > 0 else None,
               "worker_error_count": len(worker_errors),
               "worker_errors": worker_errors[:5],
               "status": "ok" if (window and not worker_errors) else "FAILED"}
        phase_results.append(res)
        _state["phases"][idx]["result"] = res
        print(f"[load] {name}: {json.dumps(res)}", flush=True)
        with _state_lock:
            write_state()

        # phase result is recorded in the table; the chart series is the rolling
        # live rate from the telemetry thread, so nothing is double-counted here.
        with _state_lock:
            _push("ts", time.time())
    reqs.close()
    errlog.close()
    stop_tel.set()
    summary = {"started_utc": _state["started_utc"],
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "phases": phase_results, "totals": _state["totals"],
               "shapes": _state["shapes"],
               "method": "openai_portable_concurrency_soak (non-streaming; e2e per PROCEDURES §4)"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    with _state_lock:
        _state["phase"] = "DONE"
        write_state()
    bad = [p["phase"] for p in phase_results if p.get("status") != "ok"]
    print("[load] DONE " + json.dumps(summary["totals"]) +
          ("  FAILED PHASES: " + ", ".join(bad) if bad else "  all phases ok"), flush=True)
    # a phase that measured nothing must not be reported as a finished run
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
