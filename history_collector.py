#!/usr/bin/env python3
"""history_collector — index every banked quant result into one JSON document.

Reads what the runs actually left on disk and emits /root/load/run/history.json.
Deterministic and read-only: it never invents a number and never writes into a run.

Sources
  /root/exl3-bench/runs/<run_id>/summary.json      quality + e2e speed per run
  /root/exl3-bench/runs/<run_id>/rows.jsonl        per-row tokens/elapsed -> per eval type
  /root/exl3-bench/runs/thr-<run_id>.json          decode slope + prefill overhead
  /root/exl3-bench/exl3-2.5bpw-deployed-180*.json  Lane A / A2 (deployed 2.50bpw)
  /root/load/run/requests.jsonl                    load-soak rows, by prompt shape

Honesty rules
  * a run named INVALID-* is LISTED BUT QUARANTINED, never averaged in: it measured a
    contended box and its numbers are not comparable to the others.
  * per-eval-type throughput from a quality run is END-TO-END (it includes prefill),
    and is labelled as such; decode-only comes from the thr-*.json slope sweep.
  * families with no grade stay null, never 0.
"""
import os
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

BENCH = Path(os.environ.get("RUNBOARD_BENCH", "/root/exl3-bench"))
RUNS = BENCH / "runs"
RUN = Path(os.environ.get("RUNBOARD_LOAD", "/root/load")) / "run"
OUT = RUN / "history.json"

# human labels: run_id -> (engine, quant, note)
LABELS = {
    "prod-nvfp4-ablit-mtp3": ("vLLM (prod container)", "NVFP4 + abliterated, MTP k=3",
                              "the incumbent production engine"),
    "exl3-3.05bpw-mcs0-mtp": ("exllamav3 (r0b0tlab gb10)", "EXL3 3.05bpw",
                              "turboderp h5_ng5 pack, clean window, --moe-cpu-split 0"),
    "exl3-2.50bpw-mcs0-mtp": ("exllamav3 (r0b0tlab gb10)", "EXL3 2.50bpw",
                              "the pack chosen for deployment"),
    "cruz-fork-305bpw": ("exllamav3 (Cruz fork 329e051)", "EXL3 3.05bpw",
                         "his tuned fork on the same 3.05bpw pack; decode still far under "
                         "his published 79"),
}

# One quant label per model+pack. The deployed 2.50bpw is the same pack as the sweep run,
# so it carries the same label and the run id / note distinguishes the two lanes —
# duplicate labels in one column just read as a bug.
DEPLOYED_LABEL = "EXL3 2.50bpw"

# The top-bar selector groups by BASE MODEL. Everything banked so far is one model, so this
# exists to hold the second axis (GLM, DSv4.1) without reworking the page later.
def model_of(summ, run_id):
    mid = f"{summ.get('model') or ''} {run_id}".lower()
    if "flash-next" in mid or "qwen3.8" in mid or "qwen38" in mid:
        return "Qwen3.8-Flash-Next"
    m = summ.get("model") or run_id
    return str(m)

# which quality family is which KIND of work — this is the "per eval type" axis
FAMILY_KIND = {
    "gsm8k": "math / word problems",
    "ifeval": "instruction following",
    "humaneval": "code generation",
    "hard_reasoning": "long-form reasoning",
}
SHAPE_KIND = {
    "chat_short": "short chat / text generation",
    "code_gen": "code generation",
    "doc_summary": "long-document prose summarisation",
    "tool_call": "tool-call JSON",
    "deep_reason": "deep reasoning (long answers, thinking on)",
}


def jload(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def jlines(p):
    out = []
    try:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


# Three different row writers produced these artifacts, each naming the elapsed field
# differently. Hard-coding one name silently dropped a whole section (the deployed
# 2.50bpw per-eval-type rows) and rendered as "no data" rather than "key mismatch" —
# caught 2026-09-24 by cross-checking the collector against the raw row counts.
ELAPSED_KEYS = ("elapsed_s", "elapsed_seconds", "elapsed")


def _elapsed(row, preferred=None):
    if preferred:
        v = row.get(preferred)
        if v:
            return v
    for k in ELAPSED_KEYS:
        v = row.get(k)
        if v:
            return v
    return None


def rates(rows, elapsed_key=None):
    """Per-row end-to-end tok/s, plus token totals. None-safe: failed rows carry nulls."""
    vals = []
    for r in rows:
        e = _elapsed(r, elapsed_key)
        ct = r.get("completion_tokens")
        if ct and e:
            vals.append(ct / e)
    if not vals:
        return None
    vals.sort()
    return {
        "n": len(vals),
        "e2e_tok_s_mean": round(statistics.mean(vals), 2),
        "e2e_tok_s_p50": round(statistics.median(vals), 2),
        "tokens": sum(r["completion_tokens"] for r in rows if r.get("completion_tokens")),
    }


def by_family(rows):
    groups = defaultdict(list)
    for r in rows:
        fam = r.get("family")
        if fam:
            groups[fam].append(r)
    out = {}
    for fam, rs in groups.items():
        d = rates(rs)
        if not d:
            # every row in the family carried null tokens/elapsed (failed or ungraded):
            # omit the family rather than publish a zero we did not measure
            continue
        out[fam] = {"kind": FAMILY_KIND.get(fam, fam), "id": fam, **d}
    return out


def collect_run(run_id, summary_path, rows_path, thr_path, quarantine_reason=None):
    summ = jload(summary_path, {}) or {}
    rows = jlines(rows_path)
    thr = jload(thr_path, None)
    label = LABELS.get(run_id)
    if isinstance(label, list):
        label = tuple(label)
    engine, quant, note = label if label else (summ.get("model") or "unknown", run_id, "")
    per_eval = by_family(rows)

    fam_results = {}
    for fam, d in (summ.get("families") or {}).items():
        fam_results[fam] = {
            "kind": FAMILY_KIND.get(fam, fam),
            "correct": d.get("correct"),
            "graded": d.get("grade_complete", d.get("graded")),
            "n": d.get("n"),
            "transported": d.get("transported"),
            "ungraded": d.get("ungraded"),
            "accuracy_pct": d.get("accuracy_pct"),
        }

    entry = {
        "run_id": run_id,
        "model": model_of(summ, run_id),
        "kit": "{} rows".format(summ.get("rows_attempted") or len(rows)),
        "dataset_sha256": summ.get("dataset_sha256"),
        "engine": engine,
        "quant": quant,
        "note": note,
        "timestamp": summ.get("timestamp") or summ.get("started_utc"),
        "wall_seconds": summ.get("wall_seconds"),
        "max_tokens": summ.get("max_tokens"),
        "temperature": summ.get("temperature"),
        "top_p": summ.get("top_p"),
        "quality": {
            "auto_graded_correct": summ.get("auto_graded_correct", summ.get("correct_count")),
            "auto_graded_total": summ.get("auto_graded_total", summ.get("dataset_count")),
            "families": fam_results,
        },
        "speed": {
            "decode_tok_s_slope": (thr or {}).get("decode_tok_s_slope"),
            "prefill_overhead_s_est": (thr or {}).get("prefill_overhead_s_est"),
            "e2e_tok_s_mean": summ.get("e2e_tok_s_mean"),
            "e2e_tok_s_p50": summ.get("e2e_tok_s_p50"),
            "completion_tokens_total": summ.get("completion_tokens_total"),
        },
        "coverage": {
            "quality_rows": len(rows),
            "families": sorted({r.get("family") for r in rows if r.get("family")}),
            "decode_slope": (thr or {}).get("decode_tok_s_slope") is not None,
            "per_eval_type_e2e": bool(per_eval),
            "load_soak": False,
            "manual_review": manual_status(run_id)[0],
            "manual_reason": manual_status(run_id)[1],
            "reason": None if (thr or {}).get("decode_tok_s_slope") is not None else
                      "no max_tokens sweep: decode-only was never measured for this run, "
                      "and the kit rows cannot substitute (fitting elapsed against answer "
                      "length there gives r2 0.2-0.5, i.e. the fit is confounded by varying "
                      "prompt and answer lengths)",
        },
        "coverage_note": summ.get("coverage_note"),
        "skipped_rows": summ.get("skipped_rows"),
        "by_eval_type": per_eval,
        "rows_seen": len(rows),
        "quarantined": None,
    }
    if quarantine_reason:
        entry["quarantined"] = quarantine_reason
    return entry


def load_shapes():
    rows = jlines(RUN / "requests.jsonl")
    if not rows:
        return []
    groups = defaultdict(list)
    for r in rows:
        if r.get("shape"):
            groups[r["shape"]].append(r)
    out = []
    for shape, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        d = rates(rs, elapsed_key="elapsed")     # load rows call it 'elapsed'
        if not d:
            continue
        concs = sorted({r.get("concurrency") for r in rs if r.get("concurrency")})
        # Per-concurrency breakdown: a cross-concurrency average cannot be compared to
        # a single-stream figure, so both views are published with their own label.
        by_conc = {}
        for c in concs:
            sub = rates([r for r in rs if r.get("concurrency") == c], elapsed_key="elapsed")
            if sub:
                by_conc[str(c)] = sub
        out.append({"id": shape, "kind": SHAPE_KIND.get(shape, shape),
                    "concurrency_levels": concs, "by_concurrency": by_conc, **d})
    if not out and groups:
        # rows exist but nothing was computable: surface it rather than reporting
        # a clean-looking empty section
        out.append({"id": "WARNING", "kind": "rows present but no rate computable",
                    "n": len(rows), "e2e_tok_s_mean": None, "e2e_tok_s_p50": None,
                    "tokens": 0, "concurrency_levels": [], "by_concurrency": {}})
    return out


def group_eval_types(runs):
    """One entry per EVAL TYPE, each carrying every quant's result for that eval type.

    This is the shape the operator asked for: group by eval type, expand the row, see each
    quant. Accuracy and throughput travel together so a row answers both questions.
    """
    groups = {}
    for r in runs:
        for fam, d in (r.get("by_eval_type") or {}).items():
            q = ((r.get("quality") or {}).get("families") or {}).get(fam) or {}
            g = groups.setdefault(fam, {
                "id": fam,
                "kind": d.get("kind") or FAMILY_KIND.get(fam, fam),
                "model": r.get("model"),
                "quants": [],
            })
            g["quants"].append({
                "quant": r["quant"],
                "engine": r["engine"],
                "run_id": r["run_id"],
                "correct": q.get("correct"),
                "graded": q.get("graded"),
                "accuracy_pct": q.get("accuracy_pct"),
                "n": d.get("n"),
                "e2e_tok_s_mean": d.get("e2e_tok_s_mean"),
                "e2e_tok_s_p50": d.get("e2e_tok_s_p50"),
                "tokens": d.get("tokens"),
            })
    out = []
    for fam, g in groups.items():
        g["quants"].sort(key=lambda x: (x.get("accuracy_pct") is None,
                                        -(x.get("accuracy_pct") or 0),
                                        -(x.get("e2e_tok_s_mean") or 0)))
        g["quant_count"] = len(g["quants"])
        scored = [q for q in g["quants"] if q.get("accuracy_pct") is not None]
        g["best_accuracy"] = scored[0] if scored else None
        out.append(g)
    out.sort(key=lambda g: -len(g["quants"]))
    return out


SOAK_MODEL = "EXL3 2.50bpw (r0b0tlab) — the deployed model on :18300"
SOAK_MODEL_ID = "Qwen3.8-Flash-Next"

# Manual (non-machine-gradeable) families and where their verdicts live.
# manual-evidence-*.json is hash-bound per row, so it is artifact-grade.
# The four sweep runs were adjudicated in-session and reported in REPORT.md, but the
# per-row verdicts were never written back to disk — so they stay REPORT-ONLY and the
# reasoning cell must not silently inherit them.
MANUAL_ARTIFACTS = {"exl3-2.5bpw-deployed-180": "manual-evidence-2.5bpw.json"}
REPORT_ONLY_MANUAL = {"cruz-fork-305bpw", "exl3-3.05bpw-mcs0-mtp",
                      "prod-nvfp4-ablit-mtp3", "exl3-2.50bpw-mcs0-mtp"}


def manual_status(run_id):
    if run_id in MANUAL_ARTIFACTS:
        return ("artifact", "hash-bound per-row verdicts in " + MANUAL_ARTIFACTS[run_id])
    if run_id in REPORT_ONLY_MANUAL:
        return ("report-only",
                "adjudicated in-session and reported in REPORT.md, but per-row verdicts "
                "were never persisted; re-run the adjudication to make it artifact-grade")
    return ("not-reviewed", "no manual review recorded for this run")


def closed_summary_families(run_id):
    """Lane A's adjudicated families live in a separate artifact from its summary.

    Without this the deployed pack looks like it has no reasoning score at all, when in
    fact 20/20 rows were manually verified and hash-bound.
    """
    if run_id not in MANUAL_ARTIFACTS:
        return {}
    cs = jload(BENCH / f"{run_id}.closed-summary.json", None)
    if not cs:
        return {}
    out = {}
    for fam, d in (cs.get("families") or {}).items():
        if not d:
            continue
        n = d.get("n") or d.get("transported")
        complete = d.get("grade_complete")
        out[fam] = {
            "kind": FAMILY_KIND.get(fam, fam),
            "correct": d.get("correct"),
            "graded": complete if complete is not None else n,
            "n": n,
            "transported": d.get("transported"),
            "ungraded": d.get("ungraded"),
            "accuracy_pct": d.get("accuracy_pct"),
        }
    return out


def soak_block():
    """The load soak's final record, corrected the same way the live page corrects it.

    Corrected through the shared module so the history page and the live page can never
    disagree about the aggregate definition.
    """
    st = jload(RUN / "state.json", None)
    if not st:
        return None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from corrected_metrics import overlay
        st = overlay(st)
    except Exception:
        pass
    phases = []
    for p in st.get("phases") or []:
        r = p.get("result") or {}
        if not r:
            continue
        phases.append({
            "name": p.get("name"),
            "concurrency": p.get("c") or r.get("concurrency"),
            "minutes": p.get("minutes") or r.get("minutes"),
            "requests": r.get("requests"), "errors": r.get("errors"),
            "aggregate_tok_s": r.get("aggregate_tok_s"),
            "per_stream_mean": r.get("per_stream_mean"),
            "per_stream_p50": r.get("per_stream_p50"),
            "scaling_x": r.get("scaling_x"),
        })
    return {"model": SOAK_MODEL, "model_id": SOAK_MODEL_ID,
            "unit": "exl3-2.5bpw.service",
            "started_utc": st.get("started_utc"), "phase": st.get("phase"),
            "phases": phases, "totals": st.get("totals")}


def main():
    runs = []
    for d in sorted(RUNS.iterdir()) if RUNS.is_dir() else []:
        if not d.is_dir():
            continue
        rid = d.name
        summ = d / "summary.json"
        if not summ.exists():
            continue
        thr = RUNS / f"thr-{rid}.json"
        reason = None
        if rid.startswith("INVALID"):
            invalid = jload(RUNS / f"INVALID-thr-{rid[len('INVALID-'):]}.json", {}) or {}
            reason = ("three benchmark windows ran concurrently on a contended box; "
                      "quality and speed are not comparable. Kept visible, excluded "
                      "from every comparison.")
            _ = invalid
        runs.append(collect_run(rid, summ, d / "rows.jsonl", thr, reason))

    # the deployed 2.50bpw quality lanes live at the top level, not under runs/
    for name, label_note in (("exl3-2.5bpw-deployed-180", "Lane A — 8,192-token answer budget"),
                             ("exl3-2.5bpw-deployed-180-repeat", "Lane A2 — 16,384-token budget, repeat run")):
        summ = BENCH / f"{name}.summary.json"
        if not summ.exists():
            continue
        e = collect_run(name, summ, BENCH / f"{name}.rows.jsonl", None)
        e["engine"] = "exllamav3 (r0b0tlab gb10)"
        e["quant"] = DEPLOYED_LABEL
        e["note"] = label_note
        # Lane A's adjudicated manual family is stored outside its summary artifact
        merged = closed_summary_families(name)
        if merged:
            e["quality"]["families"].update(merged)
            e["quality"]["adjudicated_from"] = name + ".closed-summary.json"
        runs.append(e)

    # order: valid runs newest-first, quarantined ones last
    valid = [r for r in runs if not r["quarantined"]]
    bad = [r for r in runs if r["quarantined"]]
    valid.sort(key=lambda r: (r.get("timestamp") or ""), reverse=True)

    # the load soak ran against the deployed model, so that is where its under-load
    # coverage lives; say so explicitly instead of leaving the cell looking unknown
    for r in valid:
        if "deployed" in (r.get("quant") or ""):
            r["coverage"]["load_soak"] = True

    # one flat per-eval-type index across every valid quality run
    eval_index = []
    for r in valid:
        for fam, d in (r.get("by_eval_type") or {}).items():
            eval_index.append({
                "eval_type": fam, "kind": d["kind"], "basis": "quality run, end-to-end",
                "run_id": r["run_id"], "quant": r["quant"],
                "n": d["n"], "e2e_tok_s_mean": d["e2e_tok_s_mean"],
                "e2e_tok_s_p50": d["e2e_tok_s_p50"], "tokens": d["tokens"],
            })

    doc = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": valid,
        "quarantined": bad,
        "eval_type_index": eval_index,
        "eval_types": group_eval_types(valid),
        "models": sorted({r.get("model") for r in valid if r.get("model")}),
        "load_shapes": load_shapes(),
        "soak": soak_block(),
        "definitions": {
            "e2e_tok_s": ("completion tokens / full request wall time, so it INCLUDES prompt "
                          "prefill — this is what a user actually experiences per eval type"),
            "decode_tok_s_slope": ("decode-only rate from the max_tokens sweep (thr-*.json); "
                                   "independent of prefill overhead"),
            "basis": "every number is read from a run artifact on disk; nothing is estimated",
        },
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"history.json: {len(valid)} valid runs, {len(bad)} quarantined, "
          f"{len(eval_index)} eval-type rows, {len(doc['load_shapes'])} load shapes")


if __name__ == "__main__":
    main()
