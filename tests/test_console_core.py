"""console_core: the token path, atomic io, the disk gate and the job schema.

console_core is the one definition of the queue that BOTH the web process and the
dispatcher import, so every branch here is a way for the two to disagree about what a
job is. These tests pin each verdict string, not just the boolean.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import console_core as C


# ---------------------------------------------------------------- tokens

def test_read_token_returns_none_when_the_file_is_absent(sandbox):
    assert C.read_token(sandbox / "nope.token") is None


def test_read_token_returns_none_for_a_file_that_is_only_whitespace(sandbox):
    p = sandbox / "blank.token"
    p.write_text("   \n")
    assert C.read_token(p) is None


def test_read_token_strips_the_value(sandbox):
    p = sandbox / "real.token"
    p.write_text("  hunter2\n")
    assert C.read_token(p) == "hunter2"


def test_read_token_returns_none_when_the_path_is_a_directory(sandbox):
    # an unreadable token file must fail closed, not raise into the handler
    assert C.read_token(sandbox) is None


def test_console_and_hf_tokens_read_their_own_files(sandbox, monkeypatch):
    console = sandbox / "dispatch.token"
    console.write_text("C-token\n")
    hf = sandbox / "hf.token"
    hf.write_text("H-token\n")
    monkeypatch.setattr(C, "TOKEN_FILE", console)
    monkeypatch.setattr(C, "HF_TOKEN_FILE", hf)
    assert C.console_token() == "C-token"
    assert C.hf_token() == "H-token"


@pytest.mark.parametrize("header,expected", [
    (None, ""),
    ("", ""),
    ("   ", ""),
    ("deadbeef", "deadbeef"),
    ("Bearer deadbeef", "deadbeef"),
    ("bearer   deadbeef  ", "deadbeef"),
    ("BEARER deadbeef", "deadbeef"),
])
def test_bearer_extraction(header, expected):
    assert C.bearer(header) == expected


def test_token_ok_is_false_when_no_token_is_configured(sandbox, monkeypatch):
    monkeypatch.setattr(C, "TOKEN_FILE", sandbox / "missing.token")
    assert C.token_ok("Bearer anything") is False


def test_token_ok_is_false_without_a_header(sandbox, monkeypatch):
    tok = sandbox / "dispatch.token"
    tok.write_text("secret\n")
    monkeypatch.setattr(C, "TOKEN_FILE", tok)
    assert C.token_ok(None) is False
    assert C.token_ok("Bearer ") is False


def test_token_ok_accepts_the_right_token_and_refuses_a_near_miss(sandbox, monkeypatch):
    tok = sandbox / "dispatch.token"
    tok.write_text("secret\n")
    monkeypatch.setattr(C, "TOKEN_FILE", tok)
    assert C.token_ok("Bearer secret") is True
    assert C.token_ok("sercet") is False


# ---------------------------------------------------------------- io

def test_atomic_json_writes_indented_json_and_creates_parents(tmp_path):
    out = tmp_path / "deeper" / "doc.json"
    C.atomic_json(out, {"b": 1, "a": "x"})
    assert json.loads(out.read_text()) == {"b": 1, "a": "x"}
    assert out.read_text().startswith("{\n  ")          # indent=2
    assert not out.with_name(out.name + ".tmp").exists()  # tmp was renamed away


def test_atomic_json_stringifies_unserialisable_values(tmp_path):
    out = tmp_path / "doc.json"
    C.atomic_json(out, {"when": pathlib.Path("/tmp/x")})
    assert json.loads(out.read_text()) == {"when": "/tmp/x"}


def test_read_json_returns_the_document(tmp_path):
    p = tmp_path / "doc.json"
    p.write_text('{"ok": true}')
    assert C.read_json(p) == {"ok": True}


@pytest.mark.parametrize("payload", [None, "not json at all"])
def test_read_json_falls_back_to_the_default(tmp_path, payload):
    p = tmp_path / "doc.json"
    assert C.read_json(p, {"fallback": 1}) == {"fallback": 1}
    p.write_text(payload if payload is not None else "")
    assert C.read_json(p, {"fallback": 1}) == {"fallback": 1}
    assert C.read_json(p) is None


def test_disk_free_gb_uses_the_env_path_by_default(sandbox):
    assert C.disk_free_gb() > 0


def test_disk_free_gb_reports_zero_for_a_path_that_cannot_be_measured():
    assert C.disk_free_gb("/no/such/path/anywhere") == 0.0
