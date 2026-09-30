#!/usr/bin/env python3
"""Shared core for the ZGX console.

One definition of the queue, the presets, the token and the HuggingFace queries,
imported by BOTH the read-only web process and the dispatcher. Two copies of a job
schema is how a queue silently half-works: the page writes a field the executor
does not read, and the job sits in the queue forever looking healthy.

Nothing here executes a model run. The web process imports this module only to
validate a job before dropping the file; the dispatcher is the only thing that
spawns work, and it is a separate systemd unit.
"""
from __future__ import annotations

import hmac
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path

LOAD = Path(os.environ.get("RUNBOARD_LOAD", "/root/load"))
DISPATCH = LOAD / "dispatch"
QUEUE = DISPATCH / "queue"
DONE = DISPATCH / "done"
LOGDIR = DISPATCH / "log"
LOCK = DISPATCH / ".dispatch.lock"
STATUS = DISPATCH / "status.json"
HISTORY = DISPATCH / "history.json"
TOKEN_FILE = LOAD / "dispatch.token"
HF_TOKEN_FILE = LOAD / "hf.token"
MODELS = LOAD / "models.json"

ROOT_ACCESS = "root-only, 0600"

# Where the model packs live. One definition, shared on purpose: registry.py sizes a pack
# from here and the dispatcher measures a RUNNING download against it, so the two cannot
# disagree about which directory a repo name maps to. Overridable so a checkout -- and the
# test suite -- can point at a tmp tree.
HF_CACHE = Path(os.environ.get("RUNBOARD_HF_CACHE", "/root/.cache/huggingface/hub"))


def repo_cache_dir(repo: str) -> Path:
    """The hub directory that ``hf download <repo>`` writes into."""
    return HF_CACHE / ("models--" + str(repo).replace("/", "--"))


def dir_gb(path: str | Path) -> float | None:
    """Size of a tree on disk in GB, following symlinks and counting each blob once.

    HuggingFace stores a revision as symlinks into ../../blobs/, so a walker that skips
    symlinks reports 0 GB for every pack (which is how this shipped the first time and
    put a column of zeros on the page -- a zero in a size column is indistinguishable
    from a real measurement, so it must never be produced for 'cannot measure').

    Lives here rather than in registry.py because the dispatcher has to measure a running
    download with exactly these rules; registry._dir_gb is the thin alias it kept.
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


def download_progress(repo: str, expected_gb: float | None = None) -> dict:
    """How far a download has actually got, measured in BYTES on disk.

    WHY THIS EXISTS. The console used to take the last ``NN%`` it could find in the
    ``hf download`` log. That number is a PER-FILE progress bar, so a 113 GB pack of 35
    files read "20%" for hours while the seven small files finished and the twenty-two
    large shards had not started -- a number that looks like progress, is not, and was
    read as one. (Measured 2026-09-29: status.json said 20% at 7.8 GB of 113.2 GB.)

    Progress is a fact about the disk, so read the disk. ``percent`` is None -- not 0 and
    not a guess -- whenever the total is unknown or nothing has landed yet, because a
    fabricated 0% and a real 0% are indistinguishable on the page.
    """
    gb = dir_gb(repo_cache_dir(repo))
    out: dict = {"downloaded_gb": gb, "expected_gb": expected_gb, "percent": None}
    if gb is not None and expected_gb:
        try:
            out["percent"] = f"{gb / float(expected_gb) * 100:.0f}%"
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return out


# Presets are FIXED. Every one of them either is, or is a directed subset of, the
# frozen Q200v2 text-180 kit, graded by that kit's own graders. A preset is its own
# kit: smoke-20 and kit-180 must never be ranked against each other, so the run id
# carries the preset name and the page groups by kit.
#
# rows_hint is what the selection actually picks, not what the name promises --
# --limit-per-family applies the same N to each family.
PRESETS: dict[str, dict] = {
    "smoke-20": {
        "label": "smoke",
        "runner": "lite",
        "limit_per_family": 7,
        "rows_hint": 21,
        "families": "gsm8k 7, ifeval 7, hard_reasoning 7",
        "max_tokens": 8192,
        "est_minutes": 5,
        "note": "directed subset; hard_reasoning rows are written for adjudication, not auto-graded",
    },
    "quick-60": {
        "label": "quick",
        "runner": "lite",
        "limit_per_family": 20,
        "rows_hint": 60,
        "families": "gsm8k 20, ifeval 20, hard_reasoning 20",
        "max_tokens": 8192,
        "est_minutes": 15,
        "note": "directed subset; same graders as the frozen kit",
    },
    "kit-140": {
        "label": "kit 140",
        "runner": "lite",
        "limit_per_family": 0,
        "rows_hint": 140,
        "families": "gsm8k 80, ifeval 40, hard_reasoning 20",
        "max_tokens": 16384,
        "est_minutes": 45,
        "note": "this is the banked 140-row kit; humaneval is excluded by the runner (unpublished sandbox)",
    },
    "kit-180": {
        "label": "kit 180 frozen",
        "runner": "frozen",
        "rows_hint": 180,
        "families": "gsm8k 80, ifeval 40, humaneval 40, hard_reasoning 20",
        "max_tokens": 16384,
        "est_minutes": 90,
        "note": "full fidelity via the frozen runner, including the humaneval sandbox; needs the sandbox image",
    },
}

# What a run may be pointed at.
ENGINES: dict[str, dict] = {
    "current": {"label": "whatever is serving now (no switch)", "switch": None},
    "vllm-cruz": {"label": "vLLM + vllm-exl3 fork + 3.05bpw", "switch": "vllm-cruz"},
    "cruz": {"label": "Cruz fork + 3.05bpw (exllamav3)", "switch": "cruz"},
    "exl3": {"label": "stock exllamav3 + 2.50bpw", "switch": "exl3"},
    "vllm": {"label": "vLLM prod (NVFP4 + abliterated)", "switch": "vllm"},
    "tensorfold": {"label": "TensorFold v0.3.6.3 + MLX 4-bit MTP", "switch": "tensorfold"},
}

# Concurrent streams for an eval. The frozen runner multiplexes requests and states its
# own ceiling -- "workers must be between 1 and 4 for the frozen production profile" --
# so 4 is the most this box can ask for. The lite runner is serial by construction.
MAX_WORKERS = 4
# Per-runner defaults, and they differ: the dispatcher ran the frozen kit with --workers 2
# and the lite kit serially, so a single default of 2 would have refused every lite-preset
# job (smoke-20, quick-60, kit-140) the moment workers became a validated job field --
# i.e. it would have broken the console's ordinary evals while looking like a new check.
DEFAULT_WORKERS = {"frozen": 2, "lite": 1}

# The one engine this box is expected to be serving when nothing else is running.
# Every eval restores it afterwards, whatever engine the eval ran on, so the box
# cannot be left on a test engine by a finished job. Anything that ends up serving
# something else is reported as off-baseline rather than passing quietly.
#
# This was "cruz" (the exllamav3 fork). The admin chose vLLM + vllm-exl3 on the
# 3.05bpw pack as the configuration to keep live, so the baseline is that engine and
# the exllamav3 fork becomes a switchable alternative -- otherwise every run would
# "restore" a 10-minute engine load back onto the exllamav3 build the box is not
# meant to be running.
BASELINE = "vllm-cruz"

# Leave this much free after a download. The pack we run is 85 GB; an abliterated
# NVFP4 pack is ~800 GB. A button that can fill the disk is not a feature.
DISK_FLOOR_GB = 200.0
MAX_DOWNLOAD_GB = 1000.0

RUNNER_LITE = os.environ.get("RUNBOARD_RUNNER_LITE", "/root/q200_lite.py")
RUNNER_FROZEN = os.environ.get(
    "RUNBOARD_RUNNER_FROZEN",
    "/root/exl3-engine/r0b0bench/subsets/q200v2/scripts/run_quality_set.py")
RUNS_DIR = Path(os.environ.get("RUNBOARD_RUNS_DIR", "/root/exl3-bench/runs"))
# The bench root is where the runner is launched and where it drops its
# artifacts. The board reads runs/<id>/ instead (see history_collector), which is
# why the dispatcher has to move them -- both were hardcoded "/root/exl3-bench"
# in dispatcher.py, so a test could not point them anywhere else.
BENCH = Path(os.environ.get("RUNBOARD_BENCH", "/root/exl3-bench"))
SANDBOX_IMAGE = "sha256:58a0bd6b97f7001475fbe7ec8052bf2a7b0f4b7fb507097e5594c5d5b4d28644"


# ---------------------------------------------------------------- tokens

def read_token(path: Path) -> str | None:
    try:
        value = path.read_text().strip()
        return value or None
    except Exception:
        return None


def console_token() -> str | None:
    return read_token(TOKEN_FILE)


def hf_token() -> str | None:
    """Lifted out of the prod .env into a root-only file. Used for gated repos."""
    return read_token(HF_TOKEN_FILE)


def bearer(header: str | None) -> str:
    if not header:
        return ""
    header = header.strip()
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return header


def token_ok(header: str | None) -> bool:
    """Constant-time compare. Returns False when no token is configured, so a
    half-installed console fails closed rather than open."""
    want = console_token()
    got = bearer(header)
    if not want or not got:
        return False
    return hmac.compare_digest(got, want)


# ---------------------------------------------------------------- io

def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


# ---------------------------------------------------------------- disk

def disk_free_gb(path: str | None = None) -> float:
    path = path or os.environ.get("RUNBOARD_DISK_PATH", "/root")
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return 0.0


def download_allowed(size_gb: float | None) -> tuple[bool, str]:
    free = disk_free_gb()
    if size_gb is None:
        return False, "no size estimate - refusing to start a download blind"
    if size_gb > MAX_DOWNLOAD_GB:
        return False, f"{size_gb:.0f} GB exceeds the {MAX_DOWNLOAD_GB:.0f} GB per-download cap"
    if size_gb + DISK_FLOOR_GB > free:
        return False, (f"{size_gb:.0f} GB would leave {free - size_gb:.0f} GB free, "
                       f"below the {DISK_FLOOR_GB:.0f} GB floor")
    return True, f"{size_gb:.0f} GB fits ({free:.0f} GB free, keeps {free - size_gb:.0f} GB)"


# ---------------------------------------------------------------- jobs

def new_job_id(kind: str, suffix: str = "") -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    clean = "".join(c for c in suffix if c.isalnum() or c in "-_.")[:40]
    return "-".join(x for x in (stamp, kind, clean) if x)


def validate_job(job) -> tuple[bool, str]:
    if not isinstance(job, dict):
        return False, "job must be a JSON object"
    action = job.get("action")
    if action not in ("eval", "download", "serve"):
        return False, "action must be 'eval', 'download' or 'serve'"
    if action == "serve":
        if job.get("engine") not in ENGINES or job.get("engine") == "current":
            return False, f"serve needs a concrete engine (have {sorted(ENGINES)})"
    elif action == "eval":
        if job.get("preset") not in PRESETS:
            return False, f"unknown preset {job.get('preset')!r} (have {sorted(PRESETS)})"
        if job.get("engine", "current") not in ENGINES:
            return False, f"unknown engine {job.get('engine')!r} (have {sorted(ENGINES)})"
        # Refuse a request for concurrent streams that the chosen runner cannot deliver,
        # rather than accepting it and quietly running serial: a downscoped measurement
        # still publishes a number, and a number labelled "4 streams" that was measured
        # at 1 is a lie the history page would repeat forever.
        runner = PRESETS[job["preset"]]["runner"]
        workers = job.get("workers", DEFAULT_WORKERS.get(runner, 1))
        if isinstance(workers, bool) or not isinstance(workers, int) \
                or not 1 <= workers <= MAX_WORKERS:
            return False, f"workers must be an integer 1..{MAX_WORKERS} (got {workers!r})"
        if workers > 1 and runner != "frozen":
            return False, (f"workers={workers} needs a frozen preset -- the lite runner is "
                           f"serial, so {job['preset']!r} cannot carry concurrent streams")
    else:
        repo = job.get("repo")
        if not isinstance(repo, str) or "/" not in repo:
            return False, "repo must look like 'owner/name'"
        if any(c in repo for c in " \t\n;|&$`"):
            return False, "repo contains characters that do not belong in a repo id"
        ok, why = download_allowed(job.get("expected_gb"))
        if not ok:
            return False, why
    return True, "ok"


def job_public_view(job: dict) -> dict:
    """What the page is allowed to see back."""
    return {k: v for k, v in job.items() if k not in ("token",)}


# ---------------------------------------------------------------- huggingface

def hf_get(path: str, params: dict | None = None, timeout: int = 30, use_token: bool = False):
    url = "https://huggingface.co" + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(url, headers={"User-Agent": "zgx-console/1.0"})
    if use_token:
        tok = hf_token()
        if tok:
            req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def hf_search(query: str, limit: int = 12, sort: str = "downloads") -> list[dict]:
    rows = hf_get("/api/models", {"search": query, "limit": limit,
                                  "sort": sort, "direction": -1})
    out = []
    for m in rows if isinstance(rows, list) else []:
        out.append({
            "repo": m.get("id") or m.get("modelId"),
            "downloads": m.get("downloads"),
            "likes": m.get("likes"),
            # The search endpoint returns `createdAt` -- NOT `lastModified`/`updatedAt`
            # (verified against a live hit: keys are createdAt, downloads, id, likes,
            # modelId, pipeline_tag, private, tags). Labelling it "created" is the honest
            # reading; the first version asked for lastModified and filled an empty column.
            "created": (m.get("createdAt") or "")[:10],
            "gated": bool(m.get("gated")),
            "tags": [t for t in (m.get("tags") or []) if t in
                     ("exl3", "gguf", "nvfp4", "fp8", "safetensors")],
        })
    return out


def hf_revisions(repo: str) -> list[str]:
    """Every revision, straight from the source - this is the per-model variant list."""
    data = hf_get(f"/api/models/{repo}/refs")
    names = []
    for key in ("branches", "tags"):
        for item in data.get(key) or []:
            # A null (or a bare number) in the list is not a revision name. `str(item)`
            # turned null into the literal revision "None", which then appeared on the
            # page as a downloadable variant that does not exist.
            if isinstance(item, dict):
                name = item.get("name")
            else:
                name = item if isinstance(item, str) else None
            if name and name not in names:
                names.append(name)
    return names


def hf_size(repo: str, revision: str | None = None) -> dict:
    """Size BEFORE download. A repo whose weights live on revisions reports 0 bytes
    from the plain model endpoint, which is why the revision is part of the query."""
    path = f"/api/models/{repo}" + (f"/revision/{revision}" if revision else "")
    data = hf_get(path, {"blobs": "true"})
    files = [s for s in (data.get("siblings") or []) if s.get("size")]
    total = sum(s["size"] for s in files)
    return {"repo": repo, "revision": revision or (data.get("sha") or "main"),
            "files": len(files), "gb": round(total / 1e9, 2)}


def hf_plan(repo: str) -> dict:
    """Everything the download pane needs in one call: the revision list, the size of
    each, and whether the biggest one would fit."""
    revs = hf_revisions(repo)
    sizes = []
    for rev in revs[:12]:
        try:
            s = hf_size(repo, rev)
            if s["gb"] > 0:
                # Decide per revision, not just for the repo. The top-level verdict used
                # to judge only the BIGGEST revision, which would report a repo as
                # undownloadable purely because its largest variant is over the cap, even
                # when a smaller variant fits comfortably.
                ok, why = download_allowed(s["gb"])
                s["fits"] = ok
                s["verdict"] = why
                sizes.append(s)
        except Exception:
            pass
    smallest = min((s["gb"] for s in sizes), default=0.0)
    ok, why = download_allowed(smallest) if smallest else (False, "no sized revision found")
    return {"repo": repo, "revisions": revs, "sizes": sizes,
            "smallest_gb": smallest, "fits": ok, "verdict": why,
            "free_gb": round(disk_free_gb(), 1), "floor_gb": DISK_FLOOR_GB}
