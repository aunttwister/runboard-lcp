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
HF_CACHE = Path(os.environ.get("RUNBOARD_HF_CACHE", "/root/.cache/huggingface/hub"))

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
        "default": True,
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
]

PORT = 18300


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or "").strip()
    except Exception:
        return ""


def _dir_gb(path: str) -> float | None:
    """Size of a pack on disk, following symlinks and counting each blob once.

    HuggingFace stores a revision as symlinks into ../../blobs/, so a walker that skips
    symlinks reports 0 GB for every pack (which is how this shipped the first time and
    put a column of zeros on the page -- a zero in a size column is indistinguishable
    from a real measurement, so it must never be produced for 'cannot measure').
    """
    seen: set = set()
    total = 0
    try:
        for p in Path(path).rglob("*"):
            try:
                if p.is_dir() and not p.is_symlink():
                    continue
                st = p.stat()          # follows the symlink to the blob
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue           # several revisions share one blob
                seen.add(key)
                total += st.st_size
            except Exception:
                continue
        return round(total / 1e9, 1) or None
    except Exception:
        return None


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
        status = _run(["docker", "inspect", "-f", "{{.State.Status}}", "vllm-fn-tp1"])
        if status == "running":
            info["target"] = "vllm-prod"
            info["container"] = "vllm-fn-tp1"

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
