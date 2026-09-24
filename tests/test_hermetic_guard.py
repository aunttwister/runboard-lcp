"""The harness's own guard rails.

A hermetic suite is a property, not an intention: these tests pin the two guards that make
it one. ``touches_prod`` catches a production path, and the audit hook catches a test that
reaches the host machine by spawning a process or opening a socket.

The hook's recording logic is exercised here with a synthetic event. It is deliberately NOT
exercised by really spawning something -- that would be the very thing the guard exists to
prevent, and on this box spawning means touching a live inference server.
"""
from __future__ import annotations

from conftest import HOST_ACTIONS, SANDBOX, _audit_host_actions


def test_the_audit_hook_records_the_actions_that_would_leave_the_box():
    HOST_ACTIONS.clear()
    try:
        _audit_host_actions("subprocess.Popen", ())
        _audit_host_actions("socket.connect", ())
        assert HOST_ACTIONS == ["subprocess.Popen", "socket.connect"]
    finally:
        HOST_ACTIONS.clear()      # the autouse guard asserts this list is empty after a test


def test_the_audit_hook_ignores_ordinary_events():
    HOST_ACTIONS.clear()
    try:
        for event in ("open", "import", "exec", "object.__getattr__"):
            _audit_host_actions(event, ())
        assert HOST_ACTIONS == []
    finally:
        HOST_ACTIONS.clear()


def test_no_test_in_this_suite_reached_the_host_machine():
    # the autouse guard clears this before every test, so an empty list here is the
    # aggregate evidence that every subprocess/socket call in the suite was faked
    assert HOST_ACTIONS == []


def test_the_sandbox_is_a_throwaway_directory():
    assert SANDBOX.is_dir()
    assert SANDBOX.name.startswith("runboard-test-")
    assert str(SANDBOX).startswith("/tmp/")
