"""history_collector: the load-soak shapes and the per-eval-type index.

/console groups by kit, /history groups by eval type and expands to every quant. Both are
built from the same run entries, so a shape that cannot be measured must surface as a
warning rather than as an empty-looking section.
"""
from __future__ import annotations

import json

import history_collector as HC


def j(payload):
    return json.dumps(payload)


def write_shapes(run_dir, rows):
    (run_dir / "requests.jsonl").write_text("".join(j(r) + "\n" for r in rows))


def _load_row(shape, conc, tokens, elapsed, epoch):
    return {"shape": shape, "concurrency": conc, "completion_tokens": tokens,
            "elapsed": elapsed, "ts": epoch, "ok": True}


# ---------------------------------------------------------------- load_shapes

def test_load_shapes_is_empty_until_the_soak_writes_rows(run_dir):
    assert HC.load_shapes() == []


def test_load_shapes_groups_by_shape_most_requests_first(run_dir):
    write_shapes(run_dir, [
        _load_row("chat_short", 1, 100, 10.0, 1.0),
        _load_row("chat_short", 2, 300, 10.0, 1.0),
        _load_row("code_gen", 1, 200, 10.0, 1.0),
    ])
    out = HC.load_shapes()
    assert [s["id"] for s in out] == ["chat_short", "code_gen"]
    top = out[0]
    assert top["kind"] == "short chat / text generation"
    assert top["n"] == 2 and top["tokens"] == 400
    assert top["concurrency_levels"] == [1, 2]
    # a cross-concurrency average cannot be compared with a single-stream figure, so both
    # views are published: the blended one and the per-concurrency breakdown
    assert set(top["by_concurrency"]) == {"1", "2"}
    assert top["by_concurrency"]["1"] == {"n": 1, "e2e_tok_s_mean": 10.0,
                                          "e2e_tok_s_p50": 10.0, "tokens": 100}


def test_load_shapes_reads_the_key_the_load_rows_actually_use(run_dir):
    # load-soak rows call the duration "elapsed"; the shared _elapsed() chain still finds
    # a row that only carries one of the other spellings, rather than dropping the shape
    write_shapes(run_dir, [{"shape": "tool_call", "completion_tokens": 50,
                            "elapsed_s": 5.0}])
    assert HC.load_shapes()[0]["e2e_tok_s_mean"] == 10.0
    write_shapes(run_dir, [{"shape": "tool_call", "completion_tokens": 50, "elapsed": 5.0}])
    assert HC.load_shapes()[0]["e2e_tok_s_mean"] == 10.0


def test_load_shapes_omits_a_shape_with_no_computable_rate(run_dir):
    write_shapes(run_dir, [_load_row("chat_short", 1, 100, 10.0, 1.0),
                           {"shape": "deep_reason", "completion_tokens": None,
                            "elapsed": None}])
    assert [s["id"] for s in HC.load_shapes()] == ["chat_short"]


def test_load_shapes_warns_when_rows_exist_but_nothing_is_computable(run_dir):
    write_shapes(run_dir, [{"shape": "deep_reason", "completion_tokens": None,
                            "elapsed": None}])
    out = HC.load_shapes()
    assert len(out) == 1 and out[0]["id"] == "WARNING"
    assert out[0]["kind"] == "rows present but no rate computable"
    assert out[0]["n"] == 1 and out[0]["e2e_tok_s_mean"] is None


def test_load_shapes_ignores_rows_without_a_shape(run_dir):
    write_shapes(run_dir, [{"completion_tokens": 10, "elapsed": 1.0}])
    assert HC.load_shapes() == []


# ---------------------------------------------------------------- group_eval_types

def _run(run_id, quant, families, by_eval, model="Qwen3.8-Flash-Next"):
    return {"run_id": run_id, "quant": quant, "engine": "exl3", "model": model,
            "by_eval_type": by_eval, "quality": {"families": families}}


EVAL = {"gsm8k": {"kind": "math / word problems", "n": 7, "tokens": 900,
                  "e2e_tok_s_mean": 20.0, "e2e_tok_s_p50": 19.0}}


def test_group_eval_types_puts_every_quant_under_its_eval_type():
    runs = [
        _run("r1", "EXL3 2.50bpw", {"gsm8k": {"correct": 6, "graded": 7,
                                              "accuracy_pct": 85.7}}, EVAL),
        _run("r2", "NVFP4", {"gsm8k": {"correct": 7, "graded": 7, "accuracy_pct": 100.0}},
             EVAL),
    ]
    out = HC.group_eval_types(runs)
    assert len(out) == 1 and out[0]["id"] == "gsm8k"
    assert out[0]["kind"] == "math / word problems"
    assert out[0]["quant_count"] == 2
    # best accuracy first; accuracy travels with its throughput
    assert [q["quant"] for q in out[0]["quants"]] == ["NVFP4", "EXL3 2.50bpw"]
    assert out[0]["quants"][0]["accuracy_pct"] == 100.0
    assert out[0]["quants"][0]["e2e_tok_s_mean"] == 20.0
    assert out[0]["best_accuracy"]["run_id"] == "r2"


def test_group_eval_types_ranks_an_ungraded_quant_last_and_never_first():
    runs = [_run("r1", "ungraded", {"gsm8k": {}}, EVAL),
            _run("r2", "graded", {"gsm8k": {"accuracy_pct": 50.0}}, EVAL),
            _run("r3", "no-family-record", {}, EVAL)]
    out = HC.group_eval_types(runs)
    assert [q["quant"] for q in out[0]["quants"]] == ["graded", "ungraded",
                                                      "no-family-record"]
    assert out[0]["quants"][1]["accuracy_pct"] is None
    assert out[0]["best_accuracy"]["quant"] == "graded"


def test_group_eval_types_has_no_best_when_nothing_is_scored():
    out = HC.group_eval_types([_run("r1", "ungraded", {}, EVAL)])
    assert out[0]["best_accuracy"] is None


def test_group_eval_types_sorts_the_longest_group_first():
    other = {"ifeval": {"kind": "instruction following", "n": 4, "tokens": 10,
                        "e2e_tok_s_mean": 1.0, "e2e_tok_s_p50": 1.0}}
    runs = [_run("r1", "a", {}, EVAL),
            _run("r2", "b", {}, EVAL),
            _run("r3", "c", {}, other)]
    assert [g["id"] for g in HC.group_eval_types(runs)] == ["gsm8k", "ifeval"]


def test_group_eval_types_uses_the_family_name_when_the_row_has_no_kind():
    by_eval = {"new_family": {"n": 1, "tokens": 1, "e2e_tok_s_mean": 1.0,
                              "e2e_tok_s_p50": 1.0}}
    out = HC.group_eval_types([_run("r1", "a", {}, by_eval)])
    assert out[0]["kind"] == "new_family"


def test_group_eval_types_ignores_a_run_with_no_per_eval_rows():
    assert HC.group_eval_types([_run("r1", "a", {}, {})]) == []
