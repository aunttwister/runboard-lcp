"""console_core: the HuggingFace search / refs / size / plan helpers.

All four are mocked at ``urlopen``: the shapes below are the ones the live API
actually returned, including the key names spelled the way the source spells them
(``createdAt``, not ``lastModified``) -- guessing a key name is how the console once
shipped an empty column.
"""
from __future__ import annotations

import pytest

import console_core as C
from test_console_core_jobs import hf_urlopen  # noqa: F401  (fixture reuse)


def test_hf_search_maps_the_fields_the_api_returns(hf_urlopen):
    fake, _ = hf_urlopen
    fake.payload = [
        {"id": "turboderp/Qwen3.8-exl3", "downloads": 1234, "likes": 12,
         "createdAt": "2026-08-01T09:00:00.000Z", "gated": "auto",
         "tags": ["exl3", "safetensors", "region:us"]},
        {"modelId": "r0b0tlab/Qwen3.8-EXL3", "downloads": None, "likes": None,
         "tags": None},
    ]
    rows = C.hf_search("qwen3.8", limit=2)
    assert rows[0] == {"repo": "turboderp/Qwen3.8-exl3", "downloads": 1234, "likes": 12,
                       "created": "2026-08-01", "gated": True,
                       "tags": ["exl3", "safetensors"]}
    # a repo with no id falls back to modelId, and missing fields stay missing
    assert rows[1]["repo"] == "r0b0tlab/Qwen3.8-EXL3"
    assert rows[1]["created"] == "" and rows[1]["tags"] == [] and rows[1]["gated"] is False


def test_hf_search_tolerates_a_non_list_response(hf_urlopen):
    fake, _ = hf_urlopen
    fake.payload = {"error": "rate limited"}
    assert C.hf_search("anything") == []


def test_hf_revisions_merges_branches_and_tags_without_duplicates(hf_urlopen):
    fake, _ = hf_urlopen
    fake.payload = {"branches": [{"name": "main"}, {"name": "dev"}],
                    "tags": [{"name": "main"}, "3.05bpw_h5_ng5", ""]}
    assert C.hf_revisions("owner/name") == ["main", "dev", "3.05bpw_h5_ng5"]


def test_hf_revisions_skips_entries_that_are_not_names(hf_urlopen):
    # A null (or a bare number) in the list is not a revision. It used to be stringified,
    # so a null became the literal revision "None" -- a variant the page then offered to
    # download. Skipping it is the only honest reading.
    fake, _ = hf_urlopen
    fake.payload = {"tags": ["v1", None, 7, {"name": "v2"}, "", {"name": None}]}
    assert C.hf_revisions("owner/name") == ["v1", "v2"]


def test_hf_revisions_handles_absent_keys(hf_urlopen):
    fake, _ = hf_urlopen
    fake.payload = {}
    assert C.hf_revisions("owner/name") == []


def test_hf_size_sums_only_siblings_that_carry_a_size(hf_urlopen):
    fake, calls = hf_urlopen
    fake.payload = {"sha": "abc123",
                    "siblings": [{"size": 1_500_000_000}, {"size": 500_000_000},
                                 {"rfilename": "no-size-here"}]}
    got = C.hf_size("owner/name", "main")
    assert got == {"repo": "owner/name", "revision": "main", "files": 2, "gb": 2.0}
    assert "/revision/main" in calls[0].full_url and "blobs=true" in calls[0].full_url


def test_hf_size_without_a_revision_reports_the_default_branch_sha(hf_urlopen):
    fake, calls = hf_urlopen
    fake.payload = {"sha": "deadbeef", "siblings": []}
    got = C.hf_size("owner/name")
    assert got["revision"] == "deadbeef" and got["files"] == 0 and got["gb"] == 0.0
    assert "/revision/" not in calls[0].full_url


def test_hf_size_falls_back_to_main_when_even_the_sha_is_missing(hf_urlopen):
    fake, _ = hf_urlopen
    fake.payload = {}
    assert C.hf_size("owner/name")["revision"] == "main"


def test_hf_plan_sizes_each_revision_and_verdicts_each_one(monkeypatch):
    monkeypatch.setattr(C, "hf_revisions", lambda repo: ["small", "big", "zero"])
    monkeypatch.setattr(C, "disk_free_gb", lambda path=None: 500.0)

    def size(repo, rev=None):
        return {"repo": repo, "revision": rev, "files": 1,
                "gb": {"small": 80.0, "big": 400.0, "zero": 0.0}[rev]}

    monkeypatch.setattr(C, "hf_size", size)
    plan = C.hf_plan("owner/name")
    assert [s["revision"] for s in plan["sizes"]] == ["small", "big"]   # 0 GB dropped
    # each revision is judged on its own size, not on the biggest one
    assert plan["sizes"][0]["fits"] is True
    assert plan["sizes"][1]["fits"] is False
    # the verdict line is for the SMALLEST sized revision
    assert plan["smallest_gb"] == 80.0 and plan["fits"] is True
    assert plan["free_gb"] == 500.0 and plan["floor_gb"] == C.DISK_FLOOR_GB


def test_hf_plan_skips_a_revision_whose_size_query_fails(monkeypatch):
    monkeypatch.setattr(C, "hf_revisions", lambda repo: ["bad", "good"])
    monkeypatch.setattr(C, "disk_free_gb", lambda path=None: 500.0)
    monkeypatch.setattr(C, "hf_size", lambda repo, rev=None: (
        (_ for _ in ()).throw(RuntimeError("boom")) if rev == "bad"
        else {"repo": repo, "revision": rev, "files": 1, "gb": 40.0}))
    plan = C.hf_plan("owner/name")
    assert [s["revision"] for s in plan["sizes"]] == ["good"]


def test_hf_plan_says_so_when_no_revision_has_a_size(monkeypatch):
    monkeypatch.setattr(C, "hf_revisions", lambda repo: [])
    monkeypatch.setattr(C, "disk_free_gb", lambda path=None: 900.0)
    plan = C.hf_plan("owner/name")
    assert plan["sizes"] == [] and plan["smallest_gb"] == 0.0
    assert plan["fits"] is False and "no sized revision" in plan["verdict"]


def test_hf_plan_only_queries_the_first_twelve_revisions(monkeypatch):
    monkeypatch.setattr(C, "hf_revisions", lambda repo: [f"rev{i}" for i in range(20)])
    monkeypatch.setattr(C, "disk_free_gb", lambda path=None: 9000.0)
    seen = []

    def size(repo, rev=None):
        seen.append(rev)
        return {"repo": repo, "revision": rev, "files": 1, "gb": 10.0}

    monkeypatch.setattr(C, "hf_size", size)
    C.hf_plan("owner/name")
    assert len(seen) == 12


@pytest.mark.parametrize("preset", sorted(C.PRESETS))
def test_every_preset_declares_the_fields_the_dispatcher_reads(preset):
    # the dispatcher indexes these without .get(); a missing key is a KeyError at run time
    p = C.PRESETS[preset]
    for key in ("label", "runner", "max_tokens", "est_minutes", "families", "note"):
        assert key in p
    assert p["runner"] in ("lite", "frozen")
    if p["runner"] == "lite":
        assert "limit_per_family" in p
