"""history_collector: reading run artifacts without inventing a number.

Three honesty rules are pinned here: an INVALID-* run is listed but quarantined (never
averaged), a family with no grade stays absent rather than 0, and the elapsed field is read
under whichever name the artifact actually used -- hard-coding one name silently dropped a
whole section and rendered as "no data" instead of "key mismatch".
"""
from __future__ import annotations

import json

import pytest

import history_collector as HC


def j(payload):
    return json.dumps(payload)


# ---------------------------------------------------------------- readers

def test_jload_reads_json_and_falls_back(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(j({"a": 1}))
    assert HC.jload(p) == {"a": 1}
    assert HC.jload(tmp_path / "absent.json") is None
    assert HC.jload(tmp_path / "absent.json", {}) == {}
    (tmp_path / "bad.json").write_text("{oops")
    assert HC.jload(tmp_path / "bad.json", "fallback") == "fallback"


def test_jlines_skips_blank_and_corrupt_lines(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text(j({"n": 1}) + "\n\n" + "{oops\n" + j({"n": 2}) + "\n")
    assert HC.jlines(p) == [{"n": 1}, {"n": 2}]


def test_jlines_is_empty_for_an_unreadable_path(tmp_path):
    assert HC.jlines(tmp_path) == []
    assert HC.jlines(tmp_path / "absent.jsonl") == []


@pytest.mark.parametrize("row,preferred,expected", [
    ({"elapsed_s": 5, "elapsed": 9}, "elapsed", 9),
    ({"elapsed": 9}, "elapsed_s", 9),          # preferred missing -> fall through the list
    ({"elapsed_seconds": 7}, None, 7),
    ({"elapsed": 3}, None, 3),
    ({"elapsed_s": 0, "elapsed": 4}, "elapsed_s", 4),   # a zero is not a measurement
    ({"completion_tokens": 5}, None, None),
])
def test_elapsed_reads_whichever_name_the_artifact_used(row, preferred, expected):
    assert HC._elapsed(row, preferred) == expected


def test_rates_needs_tokens_and_an_elapsed_time():
    rows = [{"completion_tokens": 100, "elapsed": 10.0},      # 10 tok/s
            {"completion_tokens": 300, "elapsed": 10.0},      # 30 tok/s
            {"completion_tokens": None, "elapsed": 10.0},
            {"completion_tokens": 500, "elapsed": None}]
    got = HC.rates(rows)
    assert got == {"n": 2, "e2e_tok_s_mean": 20.0, "e2e_tok_s_p50": 20.0, "tokens": 900}


def test_rates_returns_none_when_nothing_is_computable():
    assert HC.rates([{"completion_tokens": None, "elapsed": None}]) is None
    assert HC.rates([]) is None


def test_by_family_omits_a_family_it_could_not_measure():
    rows = [{"family": "gsm8k", "completion_tokens": 100, "elapsed": 10.0},
            {"family": "humaneval", "completion_tokens": None, "elapsed": None},
            {"family": None, "completion_tokens": 1, "elapsed": 1.0}]
    out = HC.by_family(rows)
    assert set(out) == {"gsm8k"}
    assert out["gsm8k"]["kind"] == "math / word problems" and out["gsm8k"]["id"] == "gsm8k"
    assert out["gsm8k"]["e2e_tok_s_mean"] == 10.0


def test_by_family_falls_back_to_the_family_name_as_its_kind():
    out = HC.by_family([{"family": "new_thing", "completion_tokens": 10, "elapsed": 1.0}])
    assert out["new_thing"]["kind"] == "new_thing"


# ---------------------------------------------------------------- model_of

@pytest.mark.parametrize("summ", [{"model": "Qwen3.8-Flash-Next"}, {"model": "qwen38-x"},
                                  {"model": "QWEN3.8 flash-next"}])
def test_model_of_canonicalises_the_one_model_banked_so_far(summ):
    assert HC.model_of(summ, "run-1") == "Qwen3.8-Flash-Next"


def test_model_of_falls_back_to_the_summary_then_the_run_id():
    assert HC.model_of({"model": "GLM-5"}, "run-1") == "GLM-5"
    assert HC.model_of({}, "run-1") == "run-1"
