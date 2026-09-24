"""Drive the stdlib HTTP handlers without a socket.

``live_server`` and ``zgx_exporter`` both subclass ``BaseHTTPRequestHandler``, whose
constructor parses a request off ``self.rfile`` and writes the response to
``self.wfile``. Feeding it a fake connection object is enough to exercise the whole
handler -- no listener, no fixed port, no network, and no thread.

``request()`` returns ``(status, headers, body)`` so a test can assert on what a real
client would have seen.
"""
from __future__ import annotations

import io


class _Server:
    """The minimum a BaseHTTPRequestHandler touches: it only needs to exist."""

    server_name = "test"
    server_port = 0


class _Conn:
    """A fake ``socket``: reads come from ``raw``, writes land in ``sink``.

    CPython 3.11 hands the handler a ``_SocketWriter`` for ``wfile``, which calls
    ``sendall`` on the connection rather than ``makefile('wb')``, so both are provided.
    """

    def __init__(self, raw: bytes = b"", fail_write_at: int | None = None):
        self.raw = raw
        self.fail_write_at = fail_write_at
        self.writes = 0
        self.sink = io.BytesIO()

    def _count(self) -> None:
        self.writes += 1
        if self.fail_write_at == self.writes:
            raise BrokenPipeError("client went away")

    def makefile(self, mode: str = "rb", *a, **kw):
        if mode.startswith("r"):
            return io.BytesIO(self.raw)
        return _Sink(self)

    def sendall(self, data) -> None:
        self._count()
        self.sink.write(data)


class _Sink(io.BytesIO):
    """Response sink for the ``makefile('wb')`` path."""

    def __init__(self, conn: _Conn):
        super().__init__()
        self._conn = conn

    def write(self, b):  # noqa: D102
        self._conn._count()
        return super().write(b)


def parse(raw: bytes) -> tuple[int, dict, bytes]:
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split()[1])
    headers: dict[str, str] = {}
    for line in lines[1:]:
        key, _, value = line.partition(": ")
        headers[key.lower()] = value
    return status, headers, body


def request(handler_cls, method: str, path: str, body: bytes = b"",
            headers: dict | None = None, address=("127.0.0.1", 5555),
            fail_write_at: int | None = None) -> tuple[int, dict, bytes]:
    """Run one request through ``handler_cls`` and return the client-visible result."""
    hdrs = dict(headers or {})
    if body:
        hdrs.setdefault("Content-Length", str(len(body)))
    lines = [f"{method} {path} HTTP/1.1"]
    lines += [f"{k}: {v}" for k, v in hdrs.items()]
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body
    conn = _Conn(raw, fail_write_at=fail_write_at)
    handler_cls(conn, address, _Server())
    return parse(conn.sink.getvalue())
