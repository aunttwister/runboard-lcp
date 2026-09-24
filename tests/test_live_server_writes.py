"""live_server: the write gate.

Two POST routes exist, both of them only drop a job file into the queue -- the web process
executes nothing. What matters here is the gate in front of them: no token configured means
no writes at all (fail closed), a bad token is refused, and a job that fails validation is
rejected with the reason rather than queued for the dispatcher to trip over later.
"""
from __future__ import annotations

import json

import pytest

import console_core as C
import live_server
from httpkit import request

H = live_server.Handler
TOKEN = "console-secret"


@pytest.fixture
def token(sandbox):
    C.TOKEN_FILE.write_text(TOKEN + "\n")
    return TOKEN


def auth():
    return {"Authorization": "Bearer " + TOKEN}


def post(path, payload, headers=None, address=("127.0.0.1", 5555), raw=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    return request(H, "POST", path, body=body or b"", headers=headers, address=address)


# ---------------------------------------------------------------- routing / framing

def test_a_post_to_an_unknown_route_is_404():
    status, _, body = post("/api/reboot", {}, auth())
    assert status == 404 and json.loads(body) == {"error": "no such route"}


def test_a_post_without_a_body_is_400(token):
    status, _, body = request(H, "POST", "/api/dispatch", headers=auth())
    assert status == 400 and "1..65536 bytes" in json.loads(body)["error"]


def test_an_unparseable_content_length_is_treated_as_no_body(token):
    status, _, _ = request(H, "POST", "/api/dispatch",
                           headers={"Content-Length": "lots", "Authorization": "Bearer x"})
    assert status == 400


def test_an_oversized_body_is_refused_before_it_is_read(token):
    status, _, body = request(H, "POST", "/api/dispatch",
                              headers={**auth(), "Content-Length": str(live_server.MAX_BODY + 1)},
                              body=b"")
    assert status == 400 and "65536" in json.loads(body)["error"]


# ---------------------------------------------------------------- the auth gate

def test_writes_fail_closed_when_no_token_is_configured(sandbox, run_dir):
    """No token file at all must mean no writes -- not open writes."""
    status, _, body = post("/api/dispatch", {"preset": "smoke-20"}, auth())
    assert status == 503 and "no dispatch token configured" in json.loads(body)["error"]
    assert not list(C.QUEUE.glob("*.json"))
    assert "REFUSED /api/dispatch" in (C.LOAD / "dispatch/auth.log").read_text()


@pytest.mark.parametrize("header", [None, {"Authorization": "Bearer wrong"},
                                    {"Authorization": "secret"}])
def test_a_bad_or_missing_token_is_401(token, header):
    status, _, body = post("/api/dispatch", {"preset": "smoke-20"}, header)
    assert status == 401 and json.loads(body) == {"error": "unauthorized"}
    assert not list(C.QUEUE.glob("*.json"))
    assert "DENIED  /api/dispatch" in (C.LOAD / "dispatch/auth.log").read_text()


def test_a_valid_token_reads_a_broken_body_as_400(token):
    status, _, body = post("/api/dispatch", None, auth(), raw=b"{not json")
    assert status == 400 and "bad JSON" in json.loads(body)["error"]


# ---------------------------------------------------------------- queueing

def test_a_valid_eval_job_is_queued_with_the_defaults_filled_in(token):
    status, _, body = post("/api/dispatch", {"preset": "smoke-20"}, auth())
    doc = json.loads(body)
    assert status == 202 and doc["queued"] is True and doc["poll"] == "/api/dispatch"
    job = doc["job"]
    assert job["action"] == "eval" and job["engine"] == "current"
    assert job["job_id"].endswith("-eval-smoke-20")
    assert job["created_utc"].endswith("Z") and job["_queued_from"] == "127.0.0.1"
    on_disk = json.loads((C.QUEUE / f"{job['job_id']}.json").read_text())
    assert on_disk == job
    assert "QUEUED" in (C.LOAD / "dispatch/auth.log").read_text()


def test_a_queued_job_can_name_its_own_id_and_engine(token):
    status, _, body = post("/api/dispatch",
                           {"job_id": "mine", "action": "serve", "engine": "exl3"}, auth())
    assert status == 202 and json.loads(body)["job_id"] == "mine"
    assert (C.QUEUE / "mine.json").exists()


def test_a_download_job_gets_its_action_and_id_from_its_repo(token, monkeypatch):
    monkeypatch.setattr(C, "disk_free_gb", lambda path=None: 5000.0)
    status, _, body = post("/api/download",
                           {"repo": "owner/name", "expected_gb": 10.0}, auth())
    job = json.loads(body)["job"]
    assert status == 202 and job["action"] == "download"
    assert job["job_id"].endswith("-dl-owner_name")


def test_an_invalid_job_is_rejected_with_the_reason_attached(token):
    status, _, body = post("/api/dispatch", {"preset": "kit-9999"}, auth())
    doc = json.loads(body)
    assert status == 422 and "unknown preset" in doc["error"]
    assert doc["job"]["preset"] == "kit-9999"       # the page can show what it sent
    assert not list(C.QUEUE.glob("*.json"))
    assert "REJECT  /api/dispatch" in (C.LOAD / "dispatch/auth.log").read_text()


def test_an_unknown_client_is_recorded_as_a_question_mark(token):
    status, _, _ = post("/api/dispatch", {"preset": "smoke-20"}, auth(), address=None)
    assert status == 202
    log = (C.LOAD / "dispatch/auth.log").read_text()
    assert "from ?" in log


# ---------------------------------------------------------------- auth log

def test_auth_log_appends_a_utc_stamped_line(run_dir):
    live_server.auth_log("test line")
    text = (C.LOAD / "dispatch/auth.log").read_text()
    assert text.endswith(" test line\n") and text[:4].isdigit()


def test_auth_log_swallows_a_file_it_cannot_write(sandbox, monkeypatch):
    # e.g. the log path is a directory: auditing must never break a queued job
    monkeypatch.setattr(live_server, "AUTH_LOG", C.LOAD / "dispatch")
    live_server.auth_log("this cannot be written")
    assert (C.LOAD / "dispatch").is_dir()
