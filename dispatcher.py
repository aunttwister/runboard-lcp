#!/usr/bin/env python3
"""The executor. Deliberately NOT in the web process.

The page can only drop a job file. Everything that can occupy :18300, take the
default engine down or burn GPU hours happens here, in a separate systemd unit,
under an exclusive flock. A bug in the console page can therefore produce wrong
pixels and a bad job file, never a corrupted measurement.

Phase order for an eval job:
  validate -> remember what is serving -> switch (if asked) -> prove it answers
  -> run the preset -> refresh the history index -> restore the BASELINE engine

Restore is a finally-style guarantee, not a happy path: it runs after a crash in
the run, a timeout, or a switch failure.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import console_core as C  # noqa: E402

SERVE = os.environ.get("RUNBOARD_SERVE", "/root/serve.sh")
PORT = 18300
BASE_URL = f"http://127.0.0.1:{PORT}"
LOG_KEEP = 60          # lines of log tail exposed to the page
JOB_TIMEOUT = {"smoke-20": 1800, "quick-60": 5400, "kit-140": 14400, "kit-180": 28800}
DEFAULT_TIMEOUT = 14400


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run(cmd, timeout=60, env=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout or ""), (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:
        return 1, "", f"{type(exc).__name__}: {exc}"


def default_doc() -> dict:
    """The shape of status.json with nothing known yet."""
    return {"schema": "zgx.console.dispatch.v1", "state": "idle",
            "job": None, "phase": None, "started_utc": None,
            "finished_utc": None, "log_tail": [], "result": None,
            "error": None, "previous_engine": None, "restored_to": None,
            "restoring_to": None, "off_baseline": None,
            "updated_utc": now(), "queue_depth": 0}


class Status:
    """status.json is the only thing the page reads about a job. Written atomically
    on every change so a page poll never sees a half-written document."""

    def __init__(self):
        self.logs: list[str] = []
        self.doc = default_doc()
        self.doc["queue_depth"] = queue_depth()
        self.flush()

    @classmethod
    def resume(cls):
        """Continue the existing document instead of starting a blank one.

        The idle tick used to construct a fresh Status(), which reset restored_to,
        off_baseline and log_tail within 15 seconds of a job finishing. Those fields are
        the ONLY record that :18300 came back to the baseline -- a page that forgets them
        cannot answer "is the box where it should be?", which is the question the
        baseline exists to answer. The record survives until the next job starts.
        """
        obj = cls.__new__(cls)
        doc = {}
        try:
            doc = json.loads(C.STATUS.read_text())
        except Exception:
            doc = {}
        obj.doc = default_doc()
        obj.doc.update(doc)
        obj.logs = list(obj.doc.get("log_tail") or [])
        return obj

    def set(self, **kw):
        self.doc.update(kw)
        self.doc["updated_utc"] = now()
        self.doc["log_tail"] = self.logs[-LOG_KEEP:]
        C.atomic_json(C.STATUS, self.doc)

    def log(self, line: str):
        line = line.rstrip()
        if line:
            self.logs.append(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {line}")
        self.set()

    def flush(self):
        self.doc["log_tail"] = self.logs[-LOG_KEEP:]
        self.doc["updated_utc"] = now()
        C.atomic_json(C.STATUS, self.doc)


# ------------------------------------------------------------------ helpers

def queue_depth() -> int:
    try:
        return len([p for p in sorted(C.QUEUE.glob("*.json"))])
    except Exception:
        return 0


def oldest_job() -> Path | None:
    try:
        jobs = sorted(C.QUEUE.glob("*.json"))
    except Exception:
        return None
    for p in jobs:
        if not p.name.startswith("."):
            return p
    return None


def live_engine() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import registry
        return registry.live_engine()
    except Exception as exc:
        return {"target": "unknown", "error": f"{type(exc).__name__}: {exc}"}


def switch_value_for(target: str) -> str | None:
    """Map a live engine id (registry.live_engine's target) to its serve.sh token.

    Derived from the catalogue instead of a second hardcoded list. When this was a
    literal dict it was one of three separate lists naming the engines, and the vLLM
    engine actually serving :18300 was in none of them -- so "what is serving" read
    "none" in the console, in serve.sh and here, and a failed switch could not
    restore the engine it had displaced.
    """
    import registry

    for entry in registry.CATALOGUE:
        if entry["id"] == target:
            return entry.get("switch")
    return None


def busy_with_something_else() -> str | None:
    """Refuse to start while another :18300 user is mid-measurement. The soak and the
    campaign lanes write their own numbers, and two drivers on one model corrupt both."""
    for pat, label in (("load_soak[.]py", "load soak"),
                       ("run_quality_set[.]py", "frozen kit run"),
                       ("q200_lite[.]py", "preset run"),
                       ("bench_thr[.]py", "throughput sweep")):
        rc, out, _ = run(["pgrep", "-f", pat], timeout=10)
        if out.strip():
            return f"{label} is running (pid {out.split()[0]})"
    return None


def probe_generation(timeout=180) -> tuple[bool, str]:
    """A health check that only asks /v1/models will pass on a wedged server. Ask it
    to actually generate."""
    import urllib.request
    body = json.dumps({
        "model": "qwen38-flash-next-exl3",
        "messages": [{"role": "user", "content": "Reply with exactly one word: pong"}],
        "max_tokens": 8, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
    }).encode()
    req = urllib.request.Request(f"{BASE_URL}/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        txt = (d["choices"][0]["message"].get("content") or "").strip()
        return True, f"generated {len(txt)} chars in {time.time() - t0:.1f}s: {txt[:40]!r}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def log_tail(path: Path, n: int = 8) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


# ------------------------------------------------------------------ actions

def do_switch(job, st: Status) -> None:
    want = job.get("engine", "current")
    if want == "current":
        st.log("engine: whatever is serving now, no switch requested")
        return
    target = C.ENGINES[want]["switch"]
    cur = live_engine().get("target")
    if switch_value_for(cur) == target:
        st.log(f"engine: already on {want}, nothing to switch")
        return
    st.set(phase="switch")
    st.log(f"switching :{PORT} to {want} (serve.sh {target})")
    rc, out, err = run([SERVE, target], timeout=1200)
    for line in (out or "").splitlines()[-12:]:
        st.log("  " + line)
    if rc != 0:
        st.log(f"switch failed rc={rc} {err[:200]}")
        raise RuntimeError(f"switch to {want} failed rc={rc}")
    ok, why = probe_generation()
    st.log(f"generation probe: {'OK' if ok else 'FAIL'} — {why}")
    if not ok:
        raise RuntimeError(f"switched to {want} but it does not answer")


def do_eval(job, st: Status, run_id: str) -> dict:
    preset = C.PRESETS[job["preset"]]
    log = C.LOGDIR / f"{run_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    model_id = (live_engine() or {}).get("model_id") or "qwen38-flash-next-exl3"

    if preset["runner"] == "lite":
        cmd = [sys.executable, C.RUNNER_LITE, "--base-url", BASE_URL,
               "--model", model_id, "--run-id", run_id,
               "--out-dir", str(C.RUNS_DIR),
               "--limit-per-family", str(preset["limit_per_family"]),
               "--max-tokens", str(preset["max_tokens"])]
    else:
        cmd = [sys.executable, C.RUNNER_FROZEN, "--base-url", BASE_URL,
               "--run-id", run_id,
               "--set", "/root/exl3-engine/r0b0bench/subsets/q200v2/artifacts/"
                        "quality-text-180-v2.jsonl",
               "--model", model_id,
               "--max-tokens", str(preset["max_tokens"]), "--timeout", "1800",
               "--workers", "2", "--human-eval-timeout", "12",
               "--image-id", C.SANDBOX_IMAGE,
               "--profile-id", "console-dispatch",
               "--candidate-id", run_id,
               "--admission-config", "/root/exl3-bench/admission.json",
               "--chat-template-kwargs",
               '{"enable_thinking": true, "reasoning_effort": "low", "thinking": true}']

    timeout = JOB_TIMEOUT.get(job["preset"], DEFAULT_TIMEOUT)
    st.set(phase="eval")
    st.log(f"running preset {job['preset']} as {run_id} (timeout {timeout}s)")
    st.log("cmd: " + " ".join(str(c) for c in cmd))
    t0 = time.time()
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd="/root/exl3-bench")
        killed = False
        while proc.poll() is None:
            if time.time() - t0 > timeout:
                proc.terminate()
                time.sleep(10)
                if proc.poll() is None:
                    proc.kill()
                killed = True
                break
            time.sleep(10)
            rows = log_tail(log, 3)
            st.set(phase="eval", elapsed_s=round(time.time() - t0, 1),
                   rows_seen=sum(1 for l in log_tail(log, 400) if re.match(r"^\s*\[?\d+", l)))
            if rows:
                self_last = rows[-1][:160]
                # Status.log stores "[ts]   <line>" (the two-space indent is part of the
                # stored value), so the de-duplication guard has to compare against the
                # string that is actually stored. Comparing the bare line made the guard
                # always true: every 10 s poll of a running job appended the same row
                # again and the console's log tail filled with duplicates.
                if not st.logs or st.logs[-1].split("] ", 1)[-1] != "  " + self_last:
                    st.log("  " + self_last)
    rc = proc.returncode
    st.log(f"preset finished rc={rc} after {time.time() - t0:.0f}s")

    result = {"run_id": run_id, "preset": job["preset"], "rc": rc,
              "killed_by_timeout": killed, "log": str(log)}
    summary = C.RUNS_DIR / run_id / "summary.json"
    if summary.exists():
        try:
            doc = json.loads(summary.read_text())
            # Read the names the artifact actually uses. The first version of this
            # block GUESSED ("auto_graded") and reported all-null for a run that had
            # actually graded 14/14 -- a null summary is indistinguishable from a run
            # that produced nothing, so it would have hidden a successful job.
            fams = doc.get("families") or {}
            rows_n = doc.get("rows_attempted")
            result["summary"] = {
                "auto_graded": (f"{doc.get('auto_graded_correct')}/{doc.get('auto_graded_total')}"
                                if doc.get("auto_graded_total") is not None else None),
                "families": {k: f"{v.get('correct')}/{v.get('graded')} ({v.get('accuracy_pct')}%)"
                             for k, v in fams.items()},
                "e2e_tok_s_mean": doc.get("e2e_tok_s_mean"),
                "e2e_tok_s_p50": doc.get("e2e_tok_s_p50"),
                "rows_attempted": rows_n,
                "wall_seconds": doc.get("wall_seconds"),
                "completion_tokens_total": doc.get("completion_tokens_total"),
                "grader": doc.get("grader"),
                "dataset_sha256": (doc.get("dataset_sha256") or "")[:16],
                "kit": doc.get("kit") or (f"{rows_n} rows" if rows_n else None),
                "coverage_note": doc.get("coverage_note"),
                "skipped_rows": doc.get("skipped_rows"),
                "model": doc.get("model"),
            }
            st.log("summary: " + json.dumps(result["summary"])[:400])
        except Exception as exc:
            result["summary_error"] = f"{type(exc).__name__}: {exc}"
    else:
        st.log(f"no summary.json at {summary} — run may have failed before writing")
    return result


def do_download(job, st: Status) -> dict:
    repo = job["repo"]
    rev = job.get("revision") or None
    hf = None
    for cand in ("/root/dlvenv/bin/hf",
                 "/root/exl3-engine/r0b0tlab-exllamav3/.venv/bin/hf"):
        if Path(cand).exists():
            hf = cand
            break
    if hf is None:
        raise RuntimeError("no hf CLI found on this box")

    # Re-check at execution time: the queue may have sat for hours and the operator
    # may have downloaded something else in the meantime.
    ok, why = C.download_allowed(job.get("expected_gb"))
    if not ok:
        raise RuntimeError(f"refused at execution time: {why}")
    st.log(f"disk check at execution time: {why}")

    log = C.LOGDIR / f"download-{job['job_id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [hf, "download", repo]
    if rev:
        cmd += ["--revision", rev]
    env = dict(os.environ)
    tok = C.hf_token()
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    st.set(phase="download")
    st.log("cmd: " + " ".join(cmd) + (" (authenticated)" if tok else " (anonymous)"))
    t0 = time.time()
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        while proc.poll() is None:
            if time.time() - t0 > 14400:
                proc.kill()
                raise RuntimeError("download exceeded 4 h")
            time.sleep(15)
            pct = None
            for line in reversed(log_tail(log, 40)):
                m = re.findall(r"(\d{1,3})%", line)
                if m:
                    pct = m[-1]
                    break
            st.set(phase="download", elapsed_s=round(time.time() - t0, 1),
                   percent=pct, free_gb=round(C.disk_free_gb(), 1))
    rc = proc.returncode
    st.log(f"download rc={rc} after {time.time() - t0:.0f}s")
    for line in log_tail(log, 6):
        st.log("  " + line[:160])
    if rc != 0:
        raise RuntimeError(f"hf download rc={rc}")
    # make the new pack visible in the console immediately
    run([sys.executable, "/root/load/registry.py"], timeout=180)
    return {"repo": repo, "revision": rev, "rc": rc, "log": str(log),
            "free_gb_after": round(C.disk_free_gb(), 1)}


# ------------------------------------------------------------------ main

def finish(job: dict, st: Status, state: str, result=None, error=None):
    entry = {"job_id": job.get("job_id"), "action": job.get("action"),
             "preset": job.get("preset"), "engine": job.get("engine"),
             "repo": job.get("repo"), "revision": job.get("revision"),
             "state": state, "error": error, "result": result,
             "started_utc": st.doc.get("started_utc"), "finished_utc": now(),
             "previous_engine": st.doc.get("previous_engine"),
             "restored_to": st.doc.get("restored_to")}
    hist = C.read_json(C.HISTORY, {"schema": "zgx.console.dispatch_history.v1", "jobs": []})
    hist.setdefault("jobs", []).append(entry)
    hist["jobs"] = hist["jobs"][-200:]
    C.atomic_json(C.HISTORY, hist)
    st.set(state=state, phase="done" if state == "done" else st.doc.get("phase"),
           finished_utc=now(), result=result, error=error, job=C.job_public_view(job))
    if job.get("_path"):
        try:
            Path(job["_path"]).rename(C.DONE / Path(job["_path"]).name)
        except Exception as exc:
            st.log(f"could not archive job file: {exc}")


def main() -> int:
    C.QUEUE.mkdir(parents=True, exist_ok=True)
    C.DONE.mkdir(parents=True, exist_ok=True)
    C.LOGDIR.mkdir(parents=True, exist_ok=True)

    lock = open(C.LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0  # another dispatcher already owns the box

    path = oldest_job()
    if path is None:
        # Nothing queued: publish an idle status so the page is never stale, but keep the
        # record of the job that just finished -- resume() rather than a blank Status(),
        # or restored_to/off_baseline/log_tail vanish 15 s after every job.
        st = Status.resume()
        st.set(state="idle", phase=None, job=None, queue_depth=0)
        return 0

    try:
        job = json.loads(path.read_text())
    except Exception as exc:
        path.rename(C.DONE / path.name)
        st = Status.resume()
        st.set(state="failed", error=f"unreadable job file: {exc}")
        return 1

    st = Status()
    job["_path"] = str(path)
    st.set(state="running", job=C.job_public_view(job), started_utc=now(),
           phase="validate", queue_depth=queue_depth())
    st.log(f"picked job {job.get('job_id')} ({job.get('action')})")

    ok, why = C.validate_job(job)
    if not ok:
        st.log(f"validation failed: {why}")
        finish(job, st, "failed", error=f"validation: {why}")
        return 2
    st.log(f"validated: {why}")

    busy = busy_with_something_else()
    if busy:
        # Put it back and leave it queued: a measurement in flight owns the model.
        st.log(f"deferring: {busy}")
        st.set(state="queued", phase="waiting",
               error=f"deferred — {busy}")
        try:
            path.unlink()
            name = path.name
            (C.QUEUE / name).write_text(json.dumps(
                {k: v for k, v in job.items() if k != "_path"}, indent=2))
        except Exception:
            pass
        return 0

    prev = live_engine()
    st.set(previous_engine={"target": prev.get("target"),
                            "engine_build": prev.get("engine_build")})
    st.log(f"before: serving={prev.get('target')} build={prev.get('engine_build')}")

    result = None
    error = None
    explicit_serve = job["action"] == "serve"
    try:
        if explicit_serve:
            # An explicit switch is the operator naming the state they want, so it is
            # NOT undone afterwards. It still goes through the queue so it can never
            # race a run for the port.
            target = C.ENGINES[job["engine"]]["switch"]
            st.set(phase="serve")
            st.log(f"explicit switch to {job['engine']} (serve.sh {target})")
            rc, out, err = run([SERVE, target], timeout=1200)
            for line in (out or "").splitlines()[-12:]:
                st.log("  " + line)
            if rc != 0:
                raise RuntimeError(f"serve.sh {target} rc={rc} {err[:200]}")
            ok, why = probe_generation()
            st.log(f"generation probe: {'OK' if ok else 'FAIL'} - {why}")
            if not ok:
                raise RuntimeError(f"switched to {job['engine']} but it does not answer")
            result = {"switched_to": job["engine"], "target": target,
                      "probe": why, "state": live_engine()}
        else:
            do_switch(job, st)
            if job["action"] == "eval":
                run_id = f"{job['job_id']}-{job.get('engine', 'current')}"
                result = do_eval(job, st, run_id)
            else:
                result = do_download(job, st)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        st.log(f"ERROR: {error}")
    finally:
        # The operator's rule (2026-09-24): Cruz is the baseline. :18300 therefore ends
        # every eval on the BASELINE engine, not merely on whatever happened to be
        # serving when the job was queued -- an eval that ran on a test engine must not
        # be able to leave the box on it. A job may still name another engine; it gets
        # switched for the run and the baseline comes back afterwards. An explicit
        # `serve` job is the one case where the operator's stated state is left alone,
        # and then the drift is recorded and logged rather than passing silently.
        try:
            want = switch_value_for(C.BASELINE)
            cur = live_engine().get("target") or ""
            if explicit_serve:
                if switch_value_for(cur) != want:
                    st.set(off_baseline=True)
                    st.log(f"!! off-baseline: serving {cur or 'unknown'}, baseline is "
                           f"{C.BASELINE} - explicit switch left in place; "
                           f"serve.sh {C.BASELINE} to return")
                else:
                    st.set(off_baseline=False)
                    st.log(f"explicit switch, and it is the baseline ({C.BASELINE})")
            elif want and switch_value_for(cur) != want:
                st.set(phase="restore", restoring_to=C.BASELINE)
                st.log(f"restoring baseline: {C.BASELINE} (was serving {cur or 'unknown'})")
                rc, out, _ = run([SERVE, want], timeout=1200)
                st.set(restored_to={"target": C.BASELINE, "rc": rc, "ok": rc == 0,
                                    "previous": prev.get("target")},
                       off_baseline=(rc != 0))
                if rc == 0:
                    st.log(f"restored to baseline {C.BASELINE}")
                else:
                    st.log(f"!! restore rc={rc} - :{PORT} may be OFF the baseline")
            else:
                st.set(off_baseline=False)
                st.log(f"baseline already serving ({C.BASELINE}) - no restore needed")
        except Exception as exc:
            # A restore that RAISES must still surface in the field the console reads.
            # A log line alone is not a signal: `off_baseline` stayed null, the job
            # finished "done", and "the box may be off the baseline" was invisible to
            # everything except a human reading the log tail.
            st.set(off_baseline=True,
                   restored_to={"target": C.BASELINE, "rc": None, "ok": False,
                                "previous": prev.get("target"),
                                "error": type(exc).__name__})
            st.log(f"!! restore raised: {type(exc).__name__}: {exc}")


    # make the new run show up on the history page without waiting for the 5-min timer
    try:
        run(["systemctl", "start", "load-history.service"], timeout=180)
        run([sys.executable, "/root/load/registry.py"], timeout=180)
    except Exception:
        pass

    if error:
        finish(job, st, "failed", result=result, error=error)
        st.log(f"job {job.get('job_id')} FAILED")
        return 3

    # A runner that exits non-zero did not produce a usable measurement. Finishing the job
    # as "done" made a failed eval indistinguishable from a good one on every surface that
    # reads job state -- the rc lived only inside `result`, which the history table does
    # not show. The dispatcher itself did its work (it ran the job and restored the
    # baseline), so the tick still exits 0: the JOB is what failed, not the tick.
    run_rc = result.get("rc") if isinstance(result, dict) else None
    if run_rc not in (None, 0):
        finish(job, st, "failed", result=result, error=f"runner exited rc={run_rc}")
        st.log(f"job {job.get('job_id')} FAILED (runner rc={run_rc})")
        return 0

    finish(job, st, "done", result=result)
    st.log(f"job {job.get('job_id')} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
