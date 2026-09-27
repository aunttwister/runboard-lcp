#!/usr/bin/env python3
"""Does the engine on :18300 serve requests CONCURRENTLY, and what does that do to
aggregate throughput?

Answers two things the console cannot:

1. `peak_running` from the engine's own `num_requests_running` while N streams are in
   flight. If a job is already running (an eval is one stream), peak >= N+1 is direct
   proof the engine multiplexes rather than queueing.
2. aggregate tokens/s across N streams. The engine sits at ~10% of the memory bus, so the
   per-step cost is fixed-ish latency; if that is true, batching N streams into a step
   should raise aggregate throughput roughly linearly, and the console's one-job-at-a-time
   rule is costing real throughput rather than protecting it.

Prompts are DISTINCT per stream on purpose: prefix caching is enabled, so a shared prompt
would let the streams share KV and flatter the result. Token counts come from the server's
usage block -- MTP puts a whole engine step in one SSE chunk, so counting chunks under-counts.

Read-only: sends inference requests, changes nothing.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request
import re

BASE = "http://127.0.0.1:18300"

PROMPTS = [
    "Write a numbered list of twenty short facts about the planet Mars, one line each.",
    "Write a numbered list of twenty short facts about the Roman aqueducts, one line each.",
    "Write a numbered list of twenty short facts about coffee roasting, one line each.",
    "Write a numbered list of twenty short facts about Antarctic exploration, one line each.",
    "Write a numbered list of twenty short facts about bicycle gearing, one line each.",
    "Write a numbered list of twenty short facts about medieval bookbinding, one line each.",
    "Write a numbered list of twenty short facts about tide prediction, one line each.",
    "Write a numbered list of twenty short facts about lighthouse optics, one line each.",
]


def get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read().decode()


def model_id():
    return json.loads(get("/v1/models"))["data"][0]["id"]


def engine_gauges():
    try:
        txt = get("/metrics", timeout=5)
    except Exception:
        return {}
    out = {}
    for name in ("num_requests_running", "num_requests_waiting", "gpu_cache_usage_perc"):
        m = re.search(r"^vllm:%s\S*\s+([0-9.eE+-]+)$" % name, txt, re.M)
        if m:
            out[name] = float(m.group(1))
    # Engine-WIDE generation tokens, summed over the labelled series. This is the counter
    # that answers "did total output rise, or did my streams just take a share of a fixed
    # budget?" -- and subtracting my own tokens from the delta gives the OTHER stream's rate.
    toks = re.findall(r"^vllm:generation_tokens_total\S*\s+([0-9.eE+-]+)$", txt, re.M)
    if toks:
        out["generation_tokens_total"] = sum(float(t) for t in toks)
    return out


def one(model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    dt = time.time() - t0
    return (d.get("usage") or {}).get("completion_tokens", 0), dt


def condition(model, n, max_tokens):
    """Fire n streams at once; sample the engine's own view while they are in flight."""
    res, errs = [None] * n, []
    peak = [0.0]
    peak_wait = [0.0]
    base_running = [None]
    stop = threading.Event()
    gt0 = [engine_gauges().get("generation_tokens_total")]

    def sampler():
        while not stop.is_set():
            g = engine_gauges()
            if "num_requests_running" in g:
                if base_running[0] is None:
                    base_running[0] = g["num_requests_running"]
                peak[0] = max(peak[0], g["num_requests_running"])
            peak_wait[0] = max(peak_wait[0], g.get("num_requests_waiting", 0.0))
            time.sleep(0.2)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    threads = []
    t0 = time.time()
    for i in range(n):
        def work(i=i):
            try:
                res[i] = one(model, PROMPTS[i % len(PROMPTS)], max_tokens)
            except Exception as exc:
                errs.append(f"{type(exc).__name__}: {exc}")
        t = threading.Thread(target=work)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t0
    stop.set()

    toks = sum(r[0] for r in res if r)
    lat = [r[1] for r in res if r]
    gt1 = engine_gauges().get("generation_tokens_total")
    eng_toks = None
    if gt0[0] is not None and gt1 is not None and gt1 >= gt0[0]:
        eng_toks = gt1 - gt0[0]
    return {
        "n": n,
        "wall_s": round(wall, 2),
        "tokens": toks,
        "aggregate_tps": round(toks / wall, 1) if wall else 0.0,
        "per_stream_tps": round(toks / wall / n, 1) if wall else 0.0,
        "engine_wide_tps": round(eng_toks / wall, 1) if eng_toks is not None and wall else None,
        "other_stream_tps": (round((eng_toks - toks) / wall, 1)
                             if eng_toks is not None and wall else None),
        "mean_latency_s": round(statistics.mean(lat), 2) if lat else None,
        "running_before": base_running[0],
        "peak_running": peak[0],
        "peak_waiting": peak_wait[0],
        "errors": errs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    model = model_id()
    print(f"model={model} max_tokens={args.max_tokens} label={args.label or '-'}")
    print("NOTE: any stream already running counts too -- peak_running > N proves a "
          "concurrent stream was admitted alongside the others.\n")
    rows = []
    for n in [int(x) for x in args.streams.split(",") if x.strip()]:
        r = condition(model, n, args.max_tokens)
        rows.append(r)
        print(f"  N={r['n']:<2} wall={r['wall_s']:>6.2f}s tokens={r['tokens']:<5} "
              f"aggregate={r['aggregate_tps']:>6.1f} tok/s  per-stream="
              f"{r['per_stream_tps']:>6.1f}  lat={r['mean_latency_s']}  "
              f"running {r['running_before']}->{r['peak_running']} "
              f"(waiting peak {r['peak_waiting']})  "
              f"ENGINE-WIDE={r['engine_wide_tps']} tok/s "
              f"(other stream {r['other_stream_tps']})"
              + (f"  ERRORS={r['errors']}" if r["errors"] else ""))

    base = next((r["aggregate_tps"] for r in rows if r["n"] == 1), None)
    print("\n  scaling (aggregate vs N=1):")
    for r in rows:
        if base:
            print(f"    N={r['n']:<2} x{r['aggregate_tps'] / base:.2f}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"label": args.label, "model": model, "rows": rows}, fh, indent=2)
        print(f"\n  receipt: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
