"""console_core: the disk gate, the job schema and the HuggingFace helpers.

The disk gate is the only thing standing between a console button and a full disk, and
the job schema is what the dispatcher later trusts without re-checking. Both are pinned
here, along with the HF query shapes (mocked: no test may reach huggingface.co).
"""
from __future__ import annotations

import io
import json

import pytest

import console_core as C


# ---------------------------------------------------------------- disk gate

def _free(monkeypatch, gb):
    monkeypatch.setattr(C, "disk_free_gb", lambda path=None: gb)


def test_download_allowed_refuses_an_unknown_size(monkeypatch):
    _free(monkeypatch, 5000.0)
    ok, why = C.download_allowed(None)
    assert ok is False
    assert "blind" in why


def test_download_allowed_refuses_over_the_per_download_cap(monkeypatch):
    _free(monkeypatch, 9000.0)
    ok, why = C.download_allowed(C.MAX_DOWNLOAD_GB + 1)
    assert ok is False
    assert "exceeds" in why


def test_download_allowed_refuses_when_the_floor_would_be_breached(monkeypatch):
    _free(monkeypatch, 250.0)
    ok, why = C.download_allowed(100.0)
    assert ok is False
    assert "floor" in why


def test_download_allowed_accepts_a_pack_that_fits(monkeypatch):
    _free(monkeypatch, 1000.0)
    ok, why = C.download_allowed(85.0)
    assert ok is True
    assert "fits" in why and "keeps" in why


# ---------------------------------------------------------------- job ids

def test_new_job_id_is_stamped_kind_then_sanitised_suffix():
    jid = C.new_job_id("eval", "kit-180")
    assert jid.endswith("-eval-kit-180")
    assert jid.count("-") >= 3
    assert "T" in jid and jid.endswith("Z") is False  # suffix follows the stamp


def test_new_job_id_strips_characters_that_do_not_belong():
    jid = C.new_job_id("dl", "owner/name;$(rm -rf /)")
    tail = jid.split("-dl-", 1)[1]
    assert "/" not in tail and ";" not in tail and "$" not in tail and " " not in tail


def test_new_job_id_truncates_a_very_long_suffix():
    jid = C.new_job_id("dl", "a" * 200)
    assert len(jid.split("-dl-", 1)[1]) == 40


def test_new_job_id_with_no_suffix_is_stamp_and_kind():
    assert C.new_job_id("serve").endswith("-serve")


# ---------------------------------------------------------------- job schema

@pytest.mark.parametrize("job,reason", [
    ("not a dict", "JSON object"),
    ({}, "action must be"),
    ({"action": "rm"}, "action must be"),
    ({"action": "serve"}, "concrete engine"),
    ({"action": "serve", "engine": "current"}, "concrete engine"),
    ({"action": "serve", "engine": "nope"}, "concrete engine"),
    ({"action": "eval", "preset": "kit-999"}, "unknown preset"),
    ({"action": "eval", "preset": "quick-60", "engine": "nope"}, "unknown engine"),
    ({"action": "download"}, "owner/name"),
    ({"action": "download", "repo": 42}, "owner/name"),
    ({"action": "download", "repo": "noslash"}, "owner/name"),
    ({"action": "download", "repo": "owner/name;rm"}, "do not belong"),
])
def test_validate_job_rejections(job, reason):
    ok, why = C.validate_job(job)
    assert ok is False
    assert reason in why


def test_validate_job_refuses_a_download_that_fails_the_disk_gate(monkeypatch):
    _free(monkeypatch, 100.0)
    ok, why = C.validate_job({"action": "download", "repo": "owner/name",
                              "expected_gb": 900.0})
    assert ok is False
    assert "floor" in why or "exceeds" in why


@pytest.mark.parametrize("job", [
    {"action": "eval", "preset": "smoke-20"},
    {"action": "eval", "preset": "kit-180", "engine": "exl3"},
    {"action": "serve", "engine": "vllm"},
    {"action": "download", "repo": "owner/name", "expected_gb": 10.0},
])
def test_validate_job_accepts_every_legal_shape(monkeypatch, job):
    _free(monkeypatch, 5000.0)
    assert C.validate_job(job) == (True, "ok")


def test_job_public_view_hides_the_token_only():
    job = {"job_id": "x", "action": "eval", "token": "sekrit", "preset": "smoke-20"}
    assert C.job_public_view(job) == {"job_id": "x", "action": "eval", "preset": "smoke-20"}


# ---------------------------------------------------------------- huggingface

class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def hf_urlopen(monkeypatch):
    """Capture every HF request and reply with canned JSON."""
    calls = []

    def fake(req, timeout=None):
        calls.append(req)
        payload = fake.payload
        if isinstance(payload, Exception):
            raise payload
        return _Resp(json.dumps(payload).encode())

    monkeypatch.setattr(C.urllib.request, "urlopen", fake)
    return fake, calls


def test_hf_get_builds_the_url_with_params_and_a_ua(hf_urlopen):
    fake, calls = hf_urlopen
    fake.payload = {"ok": True}
    assert C.hf_get("/api/models", {"search": "qwen 3", "limit": 5}) == {"ok": True}
    req = calls[0]
    assert req.full_url.startswith("https://huggingface.co/api/models?")
    assert "search=qwen+3" in req.full_url and "limit=5" in req.full_url
    assert req.get_header("User-agent") == "zgx-console/1.0"


def test_hf_get_omits_params_when_there_are_none(hf_urlopen):
    fake, calls = hf_urlopen
    fake.payload = []
    C.hf_get("/api/models/x/refs")
    assert "?" not in calls[0].full_url


def test_hf_get_adds_the_bearer_token_only_when_asked(sandbox, monkeypatch, hf_urlopen):
    tok = sandbox / "hf.token"
    tok.write_text("hf_secret\n")
    monkeypatch.setattr(C, "HF_TOKEN_FILE", tok)
    fake, calls = hf_urlopen
    fake.payload = []
    C.hf_get("/api/models", use_token=True)
    assert calls[-1].get_header("Authorization") == "Bearer hf_secret"
    C.hf_get("/api/models", use_token=True)  # anonymous path is covered below
    monkeypatch.setattr(C, "HF_TOKEN_FILE", sandbox / "absent.token")
    C.hf_get("/api/models", use_token=True)
    assert calls[-1].get_header("Authorization") is None
