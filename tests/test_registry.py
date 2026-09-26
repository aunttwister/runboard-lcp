"""registry: models.json -- pack sizes, the live engine, and the zero-we-must-never-print.

Two rules are pinned here. A pack size that cannot be measured must be ``null``, never 0
(a zero in a size column is indistinguishable from a real measurement, and that is exactly
what a symlink-skipping walker shipped the first time). And the serving engine must be read
from the running process, not from a unit file that may disagree with reality.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

import console_core as C
import registry


# ---------------------------------------------------------------- _run

def test_run_returns_trimmed_stdout(monkeypatch):
    class P:
        stdout = "active\n"

    monkeypatch.setattr(registry.subprocess, "run", lambda *a, **k: P())
    assert registry._run(["systemctl", "is-active", "x"]) == "active"


def test_run_returns_empty_when_the_command_is_missing(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("no such binary")

    monkeypatch.setattr(registry.subprocess, "run", boom)
    assert registry._run(["nope"]) == ""


def test_run_returns_empty_when_stdout_is_none(monkeypatch):
    class P:
        stdout = None

    monkeypatch.setattr(registry.subprocess, "run", lambda *a, **k: P())
    assert registry._run(["x"]) == ""


# ---------------------------------------------------------------- _dir_gb

def _blob(tmp_path, name="blob-1", size=3_000_000_000):
    blobs = tmp_path / "blobs"
    blobs.mkdir(exist_ok=True)
    p = blobs / name
    with p.open("wb") as fh:
        fh.truncate(size)
    return p


def test_dir_gb_follows_symlinks_into_blobs(tmp_path):
    blob = _blob(tmp_path)
    snap = tmp_path / "snapshots" / "abc"
    snap.mkdir(parents=True)
    os.symlink(blob, snap / "model.safetensors")
    assert registry._dir_gb(str(snap)) == 3.0


def test_dir_gb_counts_a_shared_blob_once(tmp_path):
    blob = _blob(tmp_path)
    for rev in ("rev-a", "rev-b"):
        d = tmp_path / rev
        d.mkdir()
        os.symlink(blob, d / "model.safetensors")
    # two revisions, one blob: a size column that double-counted would mislead the operator
    assert registry._dir_gb(str(tmp_path)) == 3.0


def test_dir_gb_ignores_directories(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "real.bin").write_bytes(b"x" * 2_000_000_000)
    assert registry._dir_gb(str(tmp_path)) == 2.0


def test_dir_gb_skips_a_broken_symlink_instead_of_failing(tmp_path):
    os.symlink(tmp_path / "gone", tmp_path / "dangling")
    _blob(tmp_path, name="real", size=1_000_000_000)
    assert registry._dir_gb(str(tmp_path)) == 1.0


def test_dir_gb_reports_none_for_an_empty_or_absent_pack(tmp_path):
    (tmp_path / "empty").mkdir()
    assert registry._dir_gb(str(tmp_path / "empty")) is None
    assert registry._dir_gb(str(tmp_path / "never-existed")) is None


def test_dir_gb_reports_none_rather_than_zero_for_a_tiny_pack(tmp_path):
    (tmp_path / "tiny").mkdir()
    (tmp_path / "tiny" / "cfg.json").write_text("{}")       # rounds to 0.0 GB
    assert registry._dir_gb(str(tmp_path / "tiny")) is None


def test_dir_gb_reports_none_when_the_walk_itself_fails(tmp_path, monkeypatch):
    def boom(self, pattern):
        raise OSError("cannot walk")

    monkeypatch.setattr(pathlib.Path, "rglob", boom)
    assert registry._dir_gb(str(tmp_path)) is None


# ---------------------------------------------------------------- live_engine

def _fake_tools(monkeypatch, active_unit=None, docker="stopped", ss=""):
    def fake_run(cmd, timeout=20):
        if cmd[0] == "systemctl":
            return "active" if active_unit == cmd[-1] else "inactive"
        if cmd[0] == "docker":
            return docker
        return ss

    monkeypatch.setattr(registry, "_run", fake_run)


def _patch_maps(monkeypatch, text=None, error=None):
    real = pathlib.Path.read_text

    def fake(self, *a, **kw):
        if str(self).startswith("/proc/") and str(self).endswith("/maps"):
            if error:
                raise error
            return text
        return real(self, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "read_text", fake)


def _patch_models(monkeypatch, payload=None, error=None):
    import urllib.request

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def fake(req, timeout=None):
        if error:
            raise error
        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake)


SS_LINE = ('LISTEN 0 128 127.0.0.1:18300 0.0.0.0:* '
           'users:(("python3",pid=4711,fd=7))')


def test_live_engine_reads_the_unit_the_pid_and_the_build(monkeypatch, tmp_path):
    _fake_tools(monkeypatch, active_unit="exl3-cruz-fork.service", ss=SS_LINE)
    _patch_maps(monkeypatch, text=f"/usr/lib/x/cruz-exllamav3_ext-1.2.so {tmp_path}")
    _patch_models(monkeypatch, {"data": [{"id": "qwen38-flash-next-exl3"}]})
    info = registry.live_engine()
    assert info["target"] == "cruz" and info["unit"] == "exl3-cruz-fork.service"
    assert info["pid"] == 4711
    assert info["engine_build"] == "CRUZ FORK (523ecd3)"
    assert info["healthy"] is True and info["model_id"] == "qwen38-flash-next-exl3"


def test_live_engine_resolves_the_vllm_engine_that_serves_the_port(monkeypatch):
    """Regression for the defect: the serving unit has to be IN the catalogue.

    vllm-exl3-cruz.service served :18300 while appearing in no engine list, so
    live_engine returned target "none" and the console printed "serving: none" beside a
    healthy model -- and a switch that then failed had no way to restore the engine it
    had displaced. Resolving the target is what lets the page, the baseline check and
    serve.sh name what is actually running.
    """
    _fake_tools(monkeypatch, active_unit="vllm-exl3-cruz.service")
    _patch_models(monkeypatch, error=OSError("refused"))
    info = registry.live_engine()
    assert info["target"] == "vllm-cruz"
    assert info["unit"] == "vllm-exl3-cruz.service"


def _patch_vllm(monkeypatch, version="0.29.0", version_error=False, models=None):
    """Fake both probes live_engine makes against the serving port."""
    import urllib.request

    class R:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.body

    def fake(req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        if url.endswith("/version"):
            if version_error:
                raise OSError("nothing answering on /version")
            return R(json.dumps({"version": version}).encode())
        return R(json.dumps(models or {"data": [{"id": "qwen3.8-flash-next"}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_live_engine_names_the_vllm_build_from_the_server_and_the_plugin(monkeypatch,
                                                                        tmp_path):
    """exllamav3 is named from the .so it loads; vLLM loads none, so ask the server.

    The panel showed a bare "-" for engine build beside a healthy model on the engine the
    box is meant to be running. The plugin version comes off its .dist-info directory name,
    never from importing it: importing vllm_exl3 pulls in the exllamav3 kernels and JITs
    them, which is how this was broken before.
    """
    _fake_tools(monkeypatch, active_unit="vllm-exl3-cruz.service")
    _patch_vllm(monkeypatch, version="0.29.0")
    site = tmp_path / "site-packages"
    (site / "vllm_exl3-0.5.0.dist-info").mkdir(parents=True)
    monkeypatch.setattr(registry, "VLLM_SITE", site)

    info = registry.live_engine()
    assert info["target"] == "vllm-cruz"
    assert info["engine_build"] == "vLLM 0.29.0 + vllm_exl3 0.5.0"


def test_live_engine_names_a_vllm_build_even_without_the_plugin_on_disk(monkeypatch,
                                                                       tmp_path):
    _fake_tools(monkeypatch, active_unit="vllm-exl3-cruz.service")
    _patch_vllm(monkeypatch, version="0.29.0")
    monkeypatch.setattr(registry, "VLLM_SITE", tmp_path / "not-installed")

    assert registry.live_engine()["engine_build"] == "vLLM 0.29.0"


def test_live_engine_leaves_the_build_unset_when_version_does_not_answer(monkeypatch,
                                                                        tmp_path):
    _fake_tools(monkeypatch, active_unit="vllm-exl3-cruz.service")
    _patch_vllm(monkeypatch, version_error=True)
    monkeypatch.setattr(registry, "VLLM_SITE", tmp_path)

    assert registry.live_engine()["engine_build"] is None


def test_live_engine_falls_back_to_the_docker_container(monkeypatch, tmp_path):
    _fake_tools(monkeypatch, docker="running", ss=SS_LINE)
    _patch_maps(monkeypatch, text="/usr/lib/r0b0tlab-exllamav3_ext.so")
    _patch_models(monkeypatch, error=OSError("refused"))
    info = registry.live_engine()
    assert info["target"] == "vllm-prod" and info["container"] == "vllm-fn-tp1"
    assert info["engine_build"] == "stock exllamav3 (r0b0tlab)"
    assert info["healthy"] is False and info["model_id"] is None
    assert info["pid"] == 4711                    # the listener is still identified
    assert info["unit"] is None


def test_live_engine_leaves_the_target_none_when_nothing_is_serving(monkeypatch):
    _fake_tools(monkeypatch)
    _patch_models(monkeypatch, error=OSError("refused"))
    info = registry.live_engine()
    assert info["target"] == "none" and info["engine_build"] is None


def test_live_engine_reports_an_unrecognised_extension_path_verbatim(monkeypatch, tmp_path):
    _fake_tools(monkeypatch, ss=SS_LINE)
    _patch_maps(monkeypatch, text=f"/opt/other/{tmp_path}/exllamav3_ext.so")
    _patch_models(monkeypatch, error=OSError("refused"))
    got = registry.live_engine()["engine_build"]
    assert got.endswith("exllamav3_ext.so") and "other" in got


def test_live_engine_ignores_a_so_that_is_not_the_extension(monkeypatch, tmp_path):
    _fake_tools(monkeypatch, ss=SS_LINE)
    _patch_maps(monkeypatch, text=f"/usr/lib/other.so {tmp_path}")
    _patch_models(monkeypatch, error=OSError("refused"))
    assert registry.live_engine()["engine_build"] is None


def test_live_engine_tolerates_an_unreadable_maps_file(monkeypatch):
    _fake_tools(monkeypatch, ss=SS_LINE)
    _patch_maps(monkeypatch, error=PermissionError("no /proc for you"))
    _patch_models(monkeypatch, error=OSError("refused"))
    info = registry.live_engine()
    assert info["pid"] == 4711 and info["engine_build"] is None


def test_live_engine_ignores_a_pid_it_cannot_parse(monkeypatch):
    _fake_tools(monkeypatch, ss='LISTEN 127.0.0.1:18300 users:(("py",pid=abc,fd=1))')
    _patch_models(monkeypatch, error=OSError("refused"))
    assert registry.live_engine()["pid"] is None


def test_live_engine_ignores_a_listener_on_another_port(monkeypatch):
    _fake_tools(monkeypatch, ss='LISTEN 0 128 127.0.0.1:9400 users:(("z",pid=1,fd=1))')
    _patch_models(monkeypatch, error=OSError("refused"))
    assert registry.live_engine()["pid"] is None


# ---------------------------------------------------------------- build

def test_build_describes_every_catalogue_entry(monkeypatch):
    monkeypatch.setattr(registry, "_run", lambda cmd, timeout=20: {
        "systemctl": "active" if cmd[1] == "is-active" else "enabled",
        "docker": "running",
    }.get(cmd[0], ""))
    monkeypatch.setattr(registry, "_dir_gb", lambda path: 12.3)
    monkeypatch.setattr(registry, "live_engine", lambda: {"target": "cruz"})
    C.TOKEN_FILE.write_text("t")
    C.HF_TOKEN_FILE.write_text("h")

    doc = registry.build()
    assert doc["schema"] == "zgx.console.models.v1"
    assert doc["port"] == registry.PORT and doc["baseline"] == "vllm-cruz"
    assert doc["serving"] == {"target": "cruz"}
    assert doc["generated_utc"].endswith("Z")
    assert doc["hf_token_present"] is True and doc["dispatch_token_present"] is True
    by_id = {e["id"]: e for e in doc["entries"]}
    assert by_id["cruz"]["size_gb"] == 12.3 and by_id["cruz"]["active"] is True
    assert by_id["cruz"]["enabled"] == "enabled"
    assert by_id["cruz"]["banked"]["decode_tok_s"] == 51.10
    assert by_id["vllm-prod"]["active"] is True and by_id["vllm-prod"]["enabled"] == "n/a"
    assert by_id["vllm-prod"]["kind"] == "docker"
    assert doc["disk"] == {"free_gb": doc["disk"]["free_gb"], "floor_gb": C.DISK_FLOOR_GB,
                           "hf_cache_gb": 12.3}
    assert doc["presets"]["smoke-20"] == C.PRESETS["smoke-20"]
    assert doc["engines"]["current"] == C.ENGINES["current"]


def test_build_reports_missing_tokens_as_absent(monkeypatch):
    monkeypatch.setattr(registry, "_run", lambda cmd, timeout=20: "")
    monkeypatch.setattr(registry, "_dir_gb", lambda path: None)
    monkeypatch.setattr(registry, "live_engine", lambda: {})
    doc = registry.build()
    assert doc["hf_token_present"] is False and doc["dispatch_token_present"] is False
    assert doc["entries"][0]["active"] is False and doc["entries"][0]["size_gb"] is None
