"""console_core.download_progress -- byte progress, and the difference between 0 and unknown.

A download's progress used to be scraped off the ``hf download`` log as the last ``NN%``
that appeared in it. That is a per-FILE bar: a 113 GB pack of 35 files read "20%" for hours
while the small files finished and the large shards had not started. These tests pin the
replacement -- a number taken from the disk -- and, just as importantly, pin the cases where
the honest answer is "I don't know" rather than a number.
"""
from __future__ import annotations

import console_core as C


def test_repo_cache_dir_maps_a_repo_name_to_the_hub_layout():
    assert C.repo_cache_dir("owner/name") == C.HF_CACHE / "models--owner--name"
    assert C.repo_cache_dir("MiaAI-Lab/Qwen3.8-Flash-Next") == \
        C.HF_CACHE / "models--MiaAI-Lab--Qwen3.8-Flash-Next"


def test_percent_is_bytes_over_the_expected_total(monkeypatch):
    monkeypatch.setattr(C, "dir_gb", lambda path: 56.6)
    assert C.download_progress("owner/name", 113.3) == {
        "downloaded_gb": 56.6, "expected_gb": 113.3, "percent": "50%"}


def test_bytes_are_reported_even_with_no_percentage(monkeypatch):
    """With no declared total there is still a fact to publish -- just not a percentage."""
    monkeypatch.setattr(C, "dir_gb", lambda path: 12.0)
    prog = C.download_progress("owner/name", None)
    assert prog["downloaded_gb"] == 12.0
    assert prog["percent"] is None


def test_nothing_on_disk_is_unknown_not_zero_percent(monkeypatch):
    """A fabricated 0% and a real 0% are indistinguishable once they are on the page."""
    monkeypatch.setattr(C, "dir_gb", lambda path: None)
    prog = C.download_progress("owner/name", 113.3)
    assert prog["downloaded_gb"] is None
    assert prog["percent"] is None


def test_a_junk_total_never_crashes_the_progress_line(monkeypatch):
    monkeypatch.setattr(C, "dir_gb", lambda path: 5.0)
    for junk in ("", "lots", 0, None, []):
        prog = C.download_progress("owner/name", junk)
        assert prog["percent"] is None, junk
        assert prog["downloaded_gb"] == 5.0, junk
