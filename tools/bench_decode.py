#!/usr/bin/env python3
"""Fixed decode benchmark for the ZGX serving engine on :18300.

Measures, against whatever is currently serving (no restart, no config change):

  * decode-only tok/s   -- tokens after the first, divided by the time between
                           the first and last token. Excludes prefill, so it is
                           comparable to the engine's own "Avg generation
                           throughput" log line and to the banked pack figures.
  * end-to-end tok/s    -- completion tokens / total request time, i.e. what the
                           eval harness and the console's throughput card report.
  * TTFT                -- prefill cost for the prompt.
  * tokens per engine step and MTP acceptance, from the engine's own
    spec_decode_* counters read before and after the window.

Greedy (temperature 0), fixed max_tokens, three prompt lengths, N repeats,
median reported. Usage:  python3 bench_decode.py [--reps 3] [--url ...]
"""
import argparse
import json
import statistics
import time
import urllib.request

# Fixed prompts. CHAT_SHORT mirrors the harness's short rows; the longer two
# measure how much prefill the same engine pays per row.
CHAT_SHORT = "What is 17 * 23? Show your working in one short paragraph."

def filler(target_words):
    """Deterministic, low-entropy context so the prompt length is reproducible."""
    unit = ("The maintenance log records routine inspections of cooling loops, "
            "power rails and interconnect fabric on the rack under test. ")
    return unit * max(1, target_words // len(unit.split()))

PROMPTS = [
    ("short", 64, CHAT_SHORT),
    ("medium", 256, "Read the following notes and summarise the key risk.\n\n" + filler(1500)),
    ("long", 256, "Read the following notes and summarise the key risk.\n\n" + filler(6000)),
]

def scrape(url, names):
    """Read named counters from the engine's Prometheus endpoint.

    vLLM emits labelled series (`name{model_name="..."} 123`), so match on the
    metric name alone and SUM across label sets -- matching `name + " "` finds
    nothing and silently yields an empty snapshot.
    """
    out = {}
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=15) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:
        return out
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        key = line.split("{")[0].split(" ")[0]
        if key not in names:
            continue
        try:
            out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return out

def stream_completion(url, prompt, max_tokens, model):
    """One streaming completion. Returns timings + token count.

    Token counting MUST come from the server's usage block: with MTP spec-decode
    vLLM packs a whole engine step's accepted tokens into one SSE chunk, so
    counting chunks measures engine STEPS, not tokens (~3.2x undercount).
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first = None
    t_last = None
    chunks = 0
    usage_tokens = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except ValueError:
                continue
            if obj.get("usage"):
                usage_tokens = obj["usage"].get("completion_tokens")
            choices = obj.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text:
                chunks += 1
                t_last = time.time()
                if t_first is None:
                    t_first = t_last
    t_end = time.time()
    return {
        "tokens": usage_tokens,
        "chunks": chunks,
        "ttft_s": (t_first - t0) if t_first else None,
        "decode_s": (t_last - t_first) if (t_first and t_last) else None,
        "total_s": t_end - t0,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18300")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--label", default="current")
    args = ap.parse_args()

    counters = ["vllm:spec_decode_num_drafts_total",
                "vllm:spec_decode_num_draft_tokens_total",
                "vllm:spec_decode_num_accepted_tokens_total",
                "vllm:iteration_tokens_total_sum",
                "vllm:iteration_tokens_total_count",
                "vllm:inter_token_latency_seconds_sum",
                "vllm:inter_token_latency_seconds_count",
                "vllm:estimated_read_bytes_per_gpu_total"]
    print(f"# label={args.label} url={args.url} reps={args.reps}")
    print(f"{'case':7} {'ttft_s':>7} {'decode_tok/s':>12} {'e2e_tok/s':>9} "
          f"{'tokens':>6} {'tok/step':>8} {'accept%':>7}")

    snap0 = scrape(args.url, counters)
    t0 = time.time()
    rows = []
    for name, mt, prompt in PROMPTS:
        runs = []
        for _ in range(args.reps):
            before = scrape(args.url, counters)
            res = stream_completion(args.url, prompt, mt, args.model)
            after = scrape(args.url, counters)

            drafts = None
            accepted = None
            if before and after:
                dd = after.get(counters[0], 0) - before.get(counters[0], 0)
                dt = after.get(counters[1], 0) - before.get(counters[1], 0)
                ac = after.get(counters[2], 0) - before.get(counters[2], 0)
                steps = dd if dd > 0 else None
                # tokens per engine step, counting the verified token itself
                if steps:
                    drafts = 1.0 + (ac / steps)
                if dt > 0:
                    accepted = 100.0 * ac / dt
            res["tok_per_step"] = drafts
            res["accept_pct"] = accepted
            r = res
            r["decode_tps"] = (r["tokens"] - 1) / r["decode_s"] if r["decode_s"] else None
            r["e2e_tps"] = r["tokens"] / r["total_s"] if r["total_s"] else None
            runs.append(r)
            time.sleep(1.0)

        def med(key):
            vals = [x[key] for x in runs if x.get(key) is not None]
            return statistics.median(vals) if vals else None

        m = med("decode_tps"); e = med("e2e_tps"); tt = med("ttft_s")
        tp = med("tok_per_step"); ac = med("accept_pct"); tk = med("tokens")
        print(f"{name:7} {tt if tt else 0:7.2f} {m if m else 0:12.1f} "
              f"{e if e else 0:9.1f} {tk or 0:6.0f} "
              f"{(tp if tp else 0):8.2f} {(ac if ac else 0):7.1f}")
        rows.append({"case": name, "ttft_s": tt, "decode_tps": m, "e2e_tps": e,
                     "tokens": tk, "tok_per_step": tp, "accept_pct": ac,
                     "runs": runs})

    # aggregate: the number a reader compares against a single-stream figure
    d = [r["decode_tps"] for r in rows if r["decode_tps"]]
    e = [r["e2e_tps"] for r in rows if r["e2e_tps"]]
    print(f"\n# median decode tok/s across cases: {statistics.median(d):.1f}" if d else "")
    print(f"# median end-to-end tok/s across cases: {statistics.median(e):.1f}" if e else "")

    # --- where the per-step time goes: the engine's own counters ---
    snap1 = scrape(args.url, counters)
    wall = time.time() - t0
    if snap0 and snap1:
        def delta(name):
            return snap1.get(name, 0.0) - snap0.get(name, 0.0)
        iters = delta("vllm:iteration_tokens_total_count")
        gen = delta("vllm:iteration_tokens_total_sum")
        lat = delta("vllm:inter_token_latency_seconds_sum")
        rbytes = delta("vllm:estimated_read_bytes_per_gpu_total")
        print("\n# --- engine-side diagnostic (whole window) ---")
        if iters:
            print(f"#   engine iterations          {iters:.0f} over {wall:.1f}s "
                  f"= {iters / wall:.1f} steps/s")
            print(f"#   tokens per iteration       {gen / iters:.2f}")
            print(f"#   mean iteration latency     {1000 * lat / iters:.1f} ms")
            if rbytes:
                print(f"#   read bytes per iteration   {rbytes / iters / 1e9:.2f} GB")
                print(f"#   achieved read bandwidth    {rbytes / wall / 1e9:.1f} GB/s "
                      f"(GB10 peak ~273 GB/s)")
        if not rbytes:
            print("#   read-bytes counter absent (engine may not expose it)")

    with open(f"/root/bench-{args.label}.json", "w") as fh:
        json.dump({"label": args.label, "rows": rows}, fh, indent=1)
    print(f"# wrote /root/bench-{args.label}.json")

if __name__ == "__main__":
    main()
