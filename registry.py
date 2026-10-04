#!/usr/bin/env python3
"""Build models.json: what the box can serve, what is serving, and what it measured.

Sizes come from the disk, serving state comes from systemd/docker/the live process,
and the banked numbers are the campaign's own published rows so the console can say
"this is what that pack measured" instead of implying a fresh number.

Run on a timer and after every job. Never writes anything the runs depend on.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import console_core as C

# The catalogue of things this box can actually put on :18300. Not a wish list:
# each entry has a launch path that exists.
# Where the model packs actually live. Overridable so a checkout can point at a tmp tree.
# One definition, in the shared core -- the dispatcher measures a running download against
# the same path, so this may not be a second copy of the rule.
HF_CACHE = C.HF_CACHE

# Where the vLLM venv keeps its distributions. Used only to read a version off a
# directory name -- see _dist_version for why this must never import the plugin.
VLLM_SITE = Path(os.environ.get("RUNBOARD_VLLM_SITE",
                                "/root/venvs/vllm-exl3/lib/python3.12/site-packages"))

CATALOGUE = [
    {
        "id": "cruz",
        "label": "Cruz fork + 3.05bpw",
        "base_model": "Qwen3.8-Flash-Next",
        "engine": "exllamav3 Cruz fork 523ecd3",
        "pack": "turboderp/Qwen3.8-Flash-Next-exl3",
        "revision": "3.05bpw_h5_ng5",
        "path": str(HF_CACHE / "models--turboderp--Qwen3.8-Flash-Next-exl3"
                    / "snapshots/69e33439ae950f17bcbe95c98f117d80f759ab6d"),
        "kind": "unit",
        "unit": "exl3-cruz-fork.service",
        "switch": "cruz",
        "model_id": "qwen38-flash-next-exl3",
        "banked": {"kit_140": "138/140 (98.6%)", "kit_140_auto": "118/120 (98.3%)",
                   "decode_tok_s": 51.10, "prefill_s": 0.831, "e2e_tok_s": 56.44},
        "default": False,
    },
    {
        "id": "vllm-cruz",
        "label": "vLLM + vllm-exl3 fork + 3.05bpw",
        "base_model": "Qwen3.8-Flash-Next",
        "engine": "vLLM + vllm-exl3 fork",
        "pack": "turboderp/Qwen3.8-Flash-Next-exl3",
        "revision": "3.05bpw_h5_ng5",
        "path": str(HF_CACHE / "models--turboderp--Qwen3.8-Flash-Next-exl3"
                    / "snapshots/69e33439ae950f17bcbe95c98f117d80f759ab6d"),
        "kind": "unit",
        "unit": "vllm-exl3-cruz.service",
        "switch": "vllm-cruz",
        "model_id": "qwen3.8-flash-next",
        # Same pack as the cruz entry above, served by a DIFFERENT engine: vLLM with the
        # vllm-exl3 plugin instead of exllamav3. It was serving :18300 through the whole
        # 2026-09-26 console work while every engine list in the toolchain -- this
        # catalogue, console_core.ENGINES and serve.sh -- still named only the three
        # exllamav3/container engines, so all of them reported "none" and the page
        # showed a false "OFF BASELINE". Rows are measured, never assumed: kit_140 is
        # absent because this engine has not run the frozen kit (smoke-20 graded 14/14).
        "banked": {"decode_tok_s": 55.0, "e2e_tok_s": 47.78},
        "default": False,
    },
    {
        "id": "exl3-2.5bpw",
        "label": "stock exllamav3 + 2.50bpw",
        "base_model": "Qwen3.8-Flash-Next",
        "engine": "exllamav3 r0b0tlab gb10",
        "pack": "r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw",
        "revision": "61a1a139ef2411ed6ed5717acfaff288511a1b08",
        "path": str(HF_CACHE / "models--r0b0tlab--Qwen3.8-Flash-Next-EXL3-2.50bpw"
                    / "snapshots/61a1a139ef2411ed6ed5717acfaff288511a1b08"),
        "kind": "unit",
        "unit": "exl3-2.5bpw.service",
        "switch": "exl3",
        "model_id": "qwen38-flash-next-exl3",
        "banked": {"kit_140": "133/140 (95.0%)", "kit_180": "172/179 graded (96.1%)",
                   "decode_tok_s": 64.58, "prefill_s": 3.34, "e2e_tok_s": 52.68},
        "default": False,
    },
    {
        "id": "vllm-prod",
        "label": "vLLM prod (NVFP4 + abliterated)",
        "base_model": "Qwen3.8-Flash-Next",
        "engine": "vLLM, container",
        "pack": "prod NVFP4 + abliterated",
        "revision": "-",
        "path": os.environ.get("RUNBOARD_VLLM_DIR", "/root/miaai-qwen3.8-single-dgx"),
        "kind": "docker",
        "unit": "vllm-fn-tp1",
        "switch": "vllm",
        "model_id": "qwen38-flash-next-exl3",
        "banked": {"kit_140": "135/140 (96.4%)", "decode_tok_s": 50.87,
                   "prefill_s": 1.88, "e2e_tok_s": 45.49},
        "default": False,
    },
    {
        "id": "tensorfold",
        "label": "TensorFold v0.3.6.3 + MLX 4-bit",
        "base_model": "Qwen3.8-Flash-Next",
        "engine": "TensorFold v0.3.6.3 (patched, CUDA/GB10)",
        "pack": "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP",
        "revision": "dadefa8066e3be900a0d148d0f5a2f4eb1cf6534",
        "path": str(HF_CACHE / "models--Vontra--Qwen3.8-Flash-Next-MLX-4bit-MTP"
                    / "snapshots/dadefa8066e3be900a0d148d0f5a2f4eb1cf6534"),
        # The recipe's own start.sh, wrapped in a systemd unit, serving :18300 like every
        # other engine so the console's switch/restore machinery needs no special case.
        # The unit is `tensorfold.service`; it carries no exllamav3 extension, so
        # live_engine() names it through the unit probe and _vllm_build() is not reached.
        # banked is deliberately empty: nothing has been measured on this engine yet, and
        # an invented row in the "what it measured last time" column is worse than a dash.
        "kind": "unit",
        "unit": "tensorfold.service",
        "switch": "tensorfold",
        "model_id": "Qwen3.8-Flash-Next",
        "banked": {},
        "default": False,
    },
    {
        # The kit that has been serving :18300 since 2026-10-03: the TensorFold engine
        # (v0.6.0 image) on the GLM-5.3-Flash EXL3 4bpw pack, tensor-parallel across BOTH
        # DGX Sparks (this box is rank 0), with the DFlash2 drafter for MTP. Discovered the
        # hard way: it was hand-started (plain `docker run`, restart policy "no") and every
        # engine list in the toolchain named only the Qwen-era engines, so "what is serving"
        # read "none" while 71 tok/s left the port.
        "id": "tensorfold-glm53",
        "label": "TensorFold v0.6.0 + GLM-5.3-Flash EXL3 4bpw (2x Spark, TP=2)",
        "base_model": "GLM-5.3-Flash",
        "engine": "TensorFold v0.6.0 (container, TP=2)",
        "pack": "Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold",
        "revision": "078455ffe6472f9a52fbc1139f58b9db2881b25c",
        "path": str(HF_CACHE / "models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold"),
        "kind": "docker",
        "unit": "glm53-flash-tf",
        "switch": "glm53",
        "model_id": "GLM-5.3-Flash-EXL3",
        "banked": {},
        "default": True,
    },
]

PORT = 18300


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or "").strip()
    except Exception:
        return ""


def _dir_gb(path: str) -> float | None:
    """Size of a pack on disk -- a thin alias for the shared walk in console_core.

    Kept as a name because the models JSON builder and its tests call it here, but the
    rules (follow symlinks into blobs, count a shared blob once, never turn 'cannot
    measure' into a 0) now live in ONE place, because the dispatcher needs the same answer
    about a directory that is still being written to.
    """
    return C.dir_gb(path)


def _dist_version(name: str) -> str | None:
    """Installed version of a distribution, read from its ``.dist-info`` directory name.

    Deliberately NOT ``import vllm_exl3``: importing the plugin pulls in the exllamav3
    kernels and JITs them for the host CPU, which is how this was broken before. The
    version is a directory name; read the directory.
    """
    for d in VLLM_SITE.glob(f"{name}-*.dist-info"):
        # "vllm_exl3-0.5.0.dist-info" -> strip the suffix, take the LAST dash field. The
        # naive split on "-" yields "0.5.0.dist" and the panel would print that verbatim.
        stem = d.name[: -len(".dist-info")]
        parts = stem.rsplit("-", 1)
        if len(parts) == 2:
            return parts[1]
    return None


def _vllm_build() -> str | None:
    """Name the vLLM build that is answering, for engines that are a python package.

    exllamav3 identifies itself through the extension ``.so`` loaded into the process, so
    the /proc maps probe above can name it. vLLM loads no such ``.so``, so that probe
    returned nothing and the panel showed a bare "-" beside a healthy model -- the same
    "the view does not say what is happening" gap as the rest of this page. Ask the server
    what it is instead. Returns None when nothing answers, so "-" stays honest rather than
    turning into a guess.
    """
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/version", timeout=6) as r:
            label = "vLLM " + json.load(r)["version"]
    except Exception:
        return None
    plugin = _dist_version("vllm_exl3")
    return f"{label} + vllm_exl3 {plugin}" if plugin else label


def live_engine() -> dict:
    """Which engine is loaded, from the running process — not from a unit file.
    The venv python is a symlink chain to /usr/bin/python3, so /proc/<pid>/exe
    cannot tell the engines apart; the loaded extension path can."""
    info = {"target": "none", "pid": None, "engine_build": None, "model_id": None,
            "healthy": False, "unit": None, "container": None}
    for entry in CATALOGUE:
        if entry["kind"] == "unit" and _run(["systemctl", "is-active", entry["unit"]]) == "active":
            info["target"] = entry["id"]
            info["unit"] = entry["unit"]
            break
    else:
        # Every docker-kind catalogue entry is probed, not one hardcoded container name:
        # a second container engine (glm53-flash-tf) was invisible here for exactly that
        # reason, and the list of things this box can serve lives in the catalogue alone.
        for entry in CATALOGUE:
            if entry["kind"] != "docker":
                continue
            status = _run(["docker", "inspect", "-f", "{{.State.Status}}", entry["unit"]])
            if status == "running":
                info["target"] = entry["id"]
                info["container"] = entry["unit"]
                break

    pid = None
    for line in _run(["ss", "-ltnp"]).splitlines():
        if f":{PORT} " in line and "pid=" in line:
            try:
                pid = int(line.split("pid=")[1].split(",")[0])
            except Exception:
                pid = None
            break
    info["pid"] = pid
    if pid:
        try:
            maps = Path(f"/proc/{pid}/maps").read_text(errors="replace")
            for tok in maps.split():
                if "exllamav3_ext" in tok and tok.endswith(".so"):
                    if "cruz-exllamav3" in tok:
                        info["engine_build"] = "CRUZ FORK (523ecd3)"
                    elif "r0b0tlab" in tok:
                        info["engine_build"] = "stock exllamav3 (r0b0tlab)"
                    else:
                        info["engine_build"] = tok
                    break
        except Exception:
            pass
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=6) as r:
            info["healthy"] = True
            info["model_id"] = json.load(r)["data"][0]["id"]
    except Exception:
        pass
    if info["engine_build"] is None and info["healthy"]:
        # Something is answering the OpenAI API and no exllamav3 extension is loaded into
        # it, so ask the server to name itself. Gated on healthy so we never probe a dark
        # port, and "None" stays available for "nothing to say".
        info["engine_build"] = _vllm_build()
    if info["engine_build"] is None and info["container"]:
        # A container engine that does not answer /version (the TensorFold engine doesn't)
        # still names itself through its own image tag -- derived from the running
        # container, never from a constant, so an image bump re-labels the panel for free.
        image = _run(["docker", "inspect", "-f", "{{.Config.Image}}", info["container"]])
        if image:
            info["engine_build"] = f"container image {image}"
    return info


def build() -> dict:
    entries = []
    for e in CATALOGUE:
        row = dict(e)
        row["size_gb"] = _dir_gb(e["path"])
        if e["kind"] == "unit":
            row["active"] = _run(["systemctl", "is-active", e["unit"]]) == "active"
            row["enabled"] = _run(["systemctl", "is-enabled", e["unit"]])
        else:
            row["active"] = _run(["docker", "inspect", "-f", "{{.State.Status}}",
                                  e["unit"]]) == "running"
            row["enabled"] = "n/a"
        entries.append(row)
    now = time.gmtime()
    return {
        "schema": "zgx.console.models.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", now),
        "port": PORT,
        "baseline": C.BASELINE,
        "serving": live_engine(),
        "entries": entries,
        "presets": {k: dict(v) for k, v in C.PRESETS.items()},
        "engines": {k: dict(v) for k, v in C.ENGINES.items()},
        "disk": {"free_gb": round(C.disk_free_gb(), 1),
                 "floor_gb": C.DISK_FLOOR_GB,
                 "hf_cache_gb": _dir_gb(str(HF_CACHE))},
        "hf_token_present": C.hf_token() is not None,
        "dispatch_token_present": C.console_token() is not None,
    }


if __name__ == "__main__":
    doc = build()
    C.atomic_json(C.MODELS, doc)
    print(f"models.json written: serving={doc['serving']['target']} "
          f"build={doc['serving']['engine_build']} free={doc['disk']['free_gb']} GB")
