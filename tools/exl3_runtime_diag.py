#!/usr/bin/env python3
"""Report the vllm-exl3 runtime policy actually selected in this environment.

Reads vllm_exl3.runtime_diagnostics() in a fresh process started with the same
env as the serving unit, so we can see which MoE kernel / ngram kernel / UVA
path the plugin picks -- rather than assuming the launch script's knobs took
effect. Per the fork README this is a PREFLIGHT, not proof of what a running
server loaded.
"""
import json
import os
import sys

KEYS = [k for k in os.environ if k.startswith(("VLLM_EXL3_", "TORCH_", "PYTORCH_"))]
print("env in effect:")
for k in sorted(KEYS):
    print(f"  {k}={os.environ[k]}")

try:
    import vllm_exl3
except Exception as e:
    print(f"\nimport vllm_exl3 FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

try:
    vllm_exl3.register()
    print("\nregister(): ok")
except Exception as e:
    print(f"\nregister() FAILED: {type(e).__name__}: {e}")

for fn in ("runtime_diagnostics", "native_moe_diagnostics",
           "prefill_policy_diagnostics"):
    f = getattr(vllm_exl3, fn, None)
    if f is None:
        continue
    try:
        print(f"\n=== {fn}() ===")
        print(json.dumps(f(), indent=1, sort_keys=True, default=str)[:3500])
    except Exception as e:
        print(f"  {fn}() failed: {type(e).__name__}: {e}")

print("\n=== top-level exports ===")
print(", ".join(n for n in dir(vllm_exl3) if not n.startswith("_"))[:700])
