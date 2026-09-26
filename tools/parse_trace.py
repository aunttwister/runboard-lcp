#!/usr/bin/env python3
"""Rank what the decode step actually spends GPU time on, from a torch chrome trace.

torch.profiler writes a chrome://tracing JSON with a "traceEvents" list. Kernel
events carry cat="kernel" and dur in microseconds. This aggregates them so the
top consumers are visible instead of guessing from launch flags.

Usage: python3 parse_trace.py [/root/prof_ext/**/*.pt.trace.json] [top_n]
"""
import glob
import json
import os
import sys
from collections import defaultdict


def find_trace(arg=None):
    if arg and os.path.isfile(arg):
        return arg
    pats = [arg] if arg else ["/root/prof_ext/**/*.json", "/root/prof_ext/*.json"]
    hits = []
    for p in pats:
        hits.extend(glob.glob(p, recursive=True))
    hits = [h for h in hits if h.endswith(".json")]
    if not hits:
        sys.exit("no trace json found")
    return max(hits, key=os.path.getsize)


def main():
    path = find_trace(sys.argv[1] if len(sys.argv) > 1 else None)
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    print(f"# trace: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    with open(path) as fh:
        doc = json.load(fh)
    events = doc.get("traceEvents") or doc

    kernels = defaultdict(lambda: [0, 0.0])   # name -> [count, total_us]
    cpu_ops = defaultdict(lambda: [0, 0.0])
    total_kernel_us = 0.0
    span = 0.0
    cats = defaultdict(int)

    for ev in events:
        cat = ev.get("cat", "")
        cats[cat] += 1
        dur = ev.get("dur") or 0.0
        if cat == "kernel":
            k = kernels[ev.get("name", "?")]
            k[0] += 1
            k[1] += dur
            total_kernel_us += dur
        elif cat in ("cpu_op", "user_annotation"):
            k = cpu_ops[ev.get("name", "?")]
            k[0] += 1
            k[1] += dur
        if ev.get("ph") == "X":
            span = max(span, (ev.get("ts") or 0) + dur)

    print(f"# event categories: {dict(sorted(cats.items(), key=lambda x: -x[1])[:8])}")
    print(f"\n# total GPU kernel time: {total_kernel_us / 1e6:.2f} s")
    if span:
        print(f"# trace span:            {span / 1e6:.2f} s")
        print(f"# GPU busy fraction:     {100 * total_kernel_us / span:.1f}%")

    print(f"\n## top {top_n} kernels by total GPU time")
    print(f"{'total_ms':>9} {'calls':>7} {'mean_us':>9}  kernel")
    for name, (cnt, tot) in sorted(kernels.items(), key=lambda x: -x[1][1])[:top_n]:
        print(f"{tot / 1000:9.1f} {cnt:7d} {tot / max(cnt, 1):9.1f}  {name[:78]}")

    print(f"\n## top 12 CPU-side ops by total time (where the host blocks)")
    for name, (cnt, tot) in sorted(cpu_ops.items(), key=lambda x: -x[1][1])[:12]:
        print(f"{tot / 1000:9.1f} {cnt:7d} {tot / max(cnt, 1):9.1f}  {name[:78]}")


if __name__ == "__main__":
    main()
