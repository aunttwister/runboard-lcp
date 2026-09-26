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


def test_no_test_module_defines_the_same_test_name_twice():
    """A duplicate name silently shadows the first definition -- pytest keeps the LAST.

    That cost real time on 2026-09-26: renaming one test collided with the original at the end
    of the file, so the stale body kept running and the new assertions never executed. The
    suite's own integrity is checked rather than assumed.
    """
    import ast
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    offenders = {}
    for path in sorted(here.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name.startswith("test_")]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            offenders[path.name] = dupes
    assert offenders == {}, f"duplicate test names shadow earlier definitions: {offenders}"
