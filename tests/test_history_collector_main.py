"""history_collector.main(): the whole index, built from a corpus of artifacts.

main() is what /history actually reads. The corpus below is shaped like the real one: two
lanes that live at the top level instead of under runs/, a quarantined run, a directory that
never produced a summary, and a run whose quant label is what marks it as the one the load
soak covered.
"""
from __future__ import annotations

import json
import shutil

import history_collector as HC


def j(payload):
    return json.dumps(payload)


FAM = {"gsm8k": {"correct": 6, "grade_complete": 7, "n": 7, "transported": 7,
                 "ungraded": 0, "accuracy_pct": 85.7}}
ROWS = [{"family": "gsm8k", "completion_tokens": 100, "elapsed_s": 10.0}]


def _summary(stamp, rows_attempted=140, model="qwen38-flash-next"):
    doc = {"rows_attempted": rows_attempted, "timestamp": stamp, "families": FAM,
           "auto_graded_correct": 118, "auto_graded_total": 120, "wall_seconds": 900,
           "dataset_sha256": "b" * 64}
    if model:
        doc["model"] = model
    return doc


def _write(root, summary, rows=ROWS, name="summary"):
    root.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        (root / f"{name}.json").write_text(j(summary))
    if rows is not None:
        (root / "rows.jsonl" if name == "summary" else f"{name}.rows.jsonl").write_text(
            "".join(j(r) + "\n" for r in rows))


def test_main_builds_the_whole_history_document(bench, run_dir, capsys, monkeypatch):
    runs = bench / "runs"
    # a banked sweep run, with its max_tokens sweep beside it
    _write(runs / "cruz-fork-305bpw", _summary("2026-09-01T00:00:00Z"))
    (runs / "thr-cruz-fork-305bpw.json").write_text(j({"decode_tok_s_slope": 51.1}))
    # a run that measured a contended box: listed, quarantined
    _write(runs / "INVALID-messy", _summary("2026-09-02T00:00:00Z"))
    (runs / "INVALID-thr-messy.json").write_text(j({"decode_tok_s_slope": 1.0}))
    # a labelled run whose quant label is what marks it as the soak's subject
    monkeypatch.setitem(HC.LABELS, "prod-deployed-180",
                        ("exllamav3", "deployed pack", "the pack the soak covered"))
    _write(runs / "prod-deployed-180", _summary("2026-09-03T00:00:00Z"))
    # a directory that never produced a summary, and a stray file
    (runs / "no-summary").mkdir()
    (runs / "notes.txt").write_text("not a run")
    # the two deployed quality lanes live at the top level, not under runs/
    (bench / "exl3-2.5bpw-deployed-180.summary.json").write_text(
        j(_summary("2026-09-04T00:00:00Z")))
    (bench / "exl3-2.5bpw-deployed-180.rows.jsonl").write_text(j(ROWS[0]) + "\n")
    (bench / "exl3-2.5bpw-deployed-180.closed-summary.json").write_text(j({
        "families": {"hard_reasoning": {"correct": 20, "n": 20, "grade_complete": None,
                                        "accuracy_pct": 100.0}}}))
    (bench / "exl3-2.5bpw-deployed-180-repeat.summary.json").write_text(
        j(_summary("2026-09-05T00:00:00Z")))
    (bench / "exl3-2.5bpw-deployed-180-repeat.rows.jsonl").write_text(j(ROWS[0]) + "\n")
    # the soak's own artifacts, in the load tree
    (run_dir / "requests.jsonl").write_text(j(
        {"shape": "chat_short", "concurrency": 1, "completion_tokens": 100,
         "elapsed": 10.0, "ts": 1.0}) + "\n")
    (run_dir / "state.json").write_text(j({"phase": "DONE", "totals": {"requests": 100},
                                           "phases": [{"name": "c1", "result": {"n": 1}}]}))

    assert HC.main() is None
    doc = json.loads((run_dir / "history.json").read_text())
    by_id = {r["run_id"]: r for r in doc["runs"]}

    assert [r["run_id"] for r in doc["runs"]] == [
        "exl3-2.5bpw-deployed-180-repeat", "exl3-2.5bpw-deployed-180",
        "prod-deployed-180", "cruz-fork-305bpw"]        # newest first
    assert [r["run_id"] for r in doc["quarantined"]] == ["INVALID-messy"]
    assert "contended box" in doc["quarantined"][0]["quarantined"]
    assert "notes.txt" not in by_id and "no-summary" not in by_id

    # the deployed lanes are relabelled onto the deployed pack
    lane_a = by_id["exl3-2.5bpw-deployed-180"]
    assert lane_a["quant"] == HC.DEPLOYED_LABEL
    assert lane_a["note"] == "Lane A — 8,192-token answer budget"
    assert lane_a["quality"]["adjudicated_from"] == (
        "exl3-2.5bpw-deployed-180.closed-summary.json")
    assert "hard_reasoning" in lane_a["quality"]["families"]
    assert "adjudicated_from" not in by_id["exl3-2.5bpw-deployed-180-repeat"]

    # the soak ran against the deployed pack, and that is where its under-load row lives
    assert by_id["prod-deployed-180"]["coverage"]["load_soak"] is True
    assert by_id["cruz-fork-305bpw"]["coverage"]["load_soak"] is False

    assert doc["models"] == ["Qwen3.8-Flash-Next"]
    assert doc["eval_type_index"][0]["basis"] == "quality run, end-to-end"
    assert {e["eval_type"] for e in doc["eval_type_index"]} == {"gsm8k"}
    assert doc["eval_types"][0]["id"] == "gsm8k"
    assert doc["load_shapes"][0]["id"] == "chat_short"
    assert doc["soak"]["phase"] == "DONE" and doc["soak"]["totals"]["requests"] == 100
    assert "nothing is estimated" in doc["definitions"]["basis"]
    assert "generated_utc" in doc

    out = capsys.readouterr().out
    assert "4 valid runs, 1 quarantined" in out
    assert "1 load shapes" in out


def test_main_writes_an_empty_history_when_nothing_is_banked(bench, run_dir, capsys):
    shutil.rmtree(bench / "runs")            # $RUNBOARD_BENCH exists, runs/ does not
    HC.main()
    doc = json.loads((run_dir / "history.json").read_text())
    assert doc["runs"] == [] and doc["quarantined"] == []
    assert doc["eval_type_index"] == [] and doc["eval_types"] == []
    assert doc["load_shapes"] == [] and doc["soak"] is None
    assert "0 valid runs, 0 quarantined" in capsys.readouterr().out
