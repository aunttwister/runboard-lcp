#!/usr/bin/env python3
"""Shared page chrome: one nav, one status strip, one look.

Three pages are served from this box and they had drifted apart. Measured before this
module existed:

  page        title                          header links (in order)
  /           "ZGX :18300 — load telemetry"  history, console
  /console    "ZGX console"                  live load view, quant history, console (this page)
  /history    "Quant results — history"      live load view, console: switch / run / download

Three names for one product, three different link sets in three different orders, exactly
one page marking where you are -- and the fact "a run is in flight" existed on one page
only, so queuing a job from the console and then looking at / showed nothing at all.

The fix is not to edit three copies consistently. It is to render the frame from ONE
definition and inject it into every page at request time, so a fourth page cannot be added
with a fourth nav.

Shared here (the frame):
  * the design tokens -- one palette, stated once, instead of three copies that had already
    drifted (`header { gap:12px }` on two pages, `gap:10px` on the third);
  * the header: product name + the view tabs, with the current page marked;
  * the status strip, filled by CHROME_JS from ``/api/live`` -- the same document the console
    reads, and the same ``job`` block that is the only component which knows about EVERY kind
    of job (the exporter's soak gauges do not: see ``live_metrics._job_block``).

NOT shared: page content styles. Each page keeps its own.

Fail loud: a page without MARKER would render with no navigation at all, so ``inject`` raises
rather than degrading silently, and the server answers 500 with the page named.
"""

MARKER = "<!--ZgX:chrome-->"
MARKER_HEAD = "<!--ZgX:chrome-head-->"
MARKER_JS = "<!--ZgX:chrome-js-->"
PRODUCT = "ZGX"

# The single definition of the view set. Order is the order rendered, on every page.
VIEWS = (
    ("/", "load telemetry"),
    ("/console", "console"),
    ("/history", "quant history"),
)

# Any path a client may use for a view -> the canonical key in VIEWS.
ALIASES = {
    "/": "/", "/index.html": "/",
    "/console": "/console", "/console.html": "/console",
    "/history": "/history", "/history.html": "/history",
}

CHROME_CSS = """
  /* ---- shared chrome: one product, one look (see ui_chrome.py). Page-specific rules
     stay in each page's own stylesheet block; anything repeated on two pages belongs
     here. Deliberately free of literal HTML tags so a naive scan of the page treats
     this block as one unit. ---- */
  :root { --bg:#0d1117; --card:#161b22; --line:#30363d; --fg:#e6edf3; --dim:#8b949e;
          --a:#58a6ff; --b:#3fb950; --c:#d29922; --d:#f85149;
          /* aliases -> house tokens, so a page may use either name */
          --accent:var(--a); --ok:var(--b); --warn:var(--c); --bad:var(--d);
          --mono:ui-monospace,SFMono-Regular,Menlo,monospace; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
  header { display:flex; flex-wrap:wrap; gap:12px; align-items:baseline;
           padding:14px 18px; border-bottom:1px solid var(--line);
           position:sticky; top:0; background:var(--bg); z-index:5; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  header a { color:var(--a); text-decoration:none; }
  header label { font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:.06em; }
  .badge { padding:2px 8px; border:1px solid var(--line); border-radius:999px; color:var(--dim); }
  .badge.on { color:var(--a); border-color:var(--a); }
  .badge.live { color:var(--b); border-color:var(--b); }
  .badge.done { color:var(--a); border-color:var(--a); }
  .badge.stale { color:var(--c); border-color:var(--c); }
  .badge.warn { color:var(--c); border-color:var(--c); }
  .ok { color:var(--b); } .warn { color:var(--c); } .bad { color:var(--d); }
  /* the view tabs inherit the header's flex layout exactly as the flat link row did */
  nav.views { display:contents; }
  /* the status strip: its own row under the tabs, on every page */
  .strip { flex-basis:100%; color:var(--dim); font-size:11.5px; }
  .strip .sep { color:var(--line); }
"""

# The strip is filled from /api/live. Kept deliberately small: it reads TWO facts that the
# product must never disagree with itself about -- the health of the two sources, and whether
# a job is active -- plus the as-of stamp so a frozen page is obvious.
CHROME_JS = """
<script>
/* Shared status strip. Same source as the console panel: /api/live, whose `job` block is
   derived from the dispatcher's own status document. Nothing here decides whether a run is
   active -- it only renders what the dispatcher says, so all three pages agree by
   construction. */
(function () {
  "use strict";
  var el = document.getElementById("zgx-strip");
  if (!el) return;
  var esc = function (s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  };
  var SEP = ' <span class="sep">·</span> ';
  function render(d) {
    var out = [], src = {}, i;
    for (i = 0; i < (d.sources || []).length; i++) src[d.sources[i].name] = d.sources[i];
    var ex = src.exporter, pr = src.prometheus;
    out.push(ex ? (ex.ok ? "exporter ok (" + esc(ex.detail) + ")"
                         : '<span class="bad">exporter unreachable</span>')
                : '<span class="bad">exporter unknown</span>');
    out.push(pr ? (pr.ok ? "history ok (" + esc(pr.detail) + ")"
                         : '<span class="warn">no history source</span>')
                : '<span class="warn">history unknown</span>');
    var j = d.job || {};
    if (j.active) {
      out.push('<span class="ok">run active: ' + esc(j.job_id || "?") + " · " +
               esc(j.preset || "?") + " · " + esc(j.phase || "?") + " · " +
               esc(j.elapsed_s === null || j.elapsed_s === undefined ? "?" : j.elapsed_s + "s") +
               "</span>");
    } else if (j.job_id) {
      out.push("no run active (last: " + esc(j.job_id) + " " + esc(j.state || "?") + ")");
    } else {
      out.push("no run active");
    }
    out.push("as of " + esc(d.fetched_utc || "?") + " · refresh 10s");
    el.innerHTML = out.join(SEP);
  }
  function poll() {
    fetch("/api/live?window=30m", { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(render)
      .catch(function (e) {
        el.innerHTML = '<span class="bad">status unavailable: ' + esc(e) +
                       '</span> -- read <a href="/api/live">/api/live</a>';
      });
  }
  poll();
  setInterval(poll, 10000);
})();
</script>
"""


def canonical(path: str) -> str:
    """Any accepted path for a view -> its key in VIEWS (unknown paths pass through)."""
    return ALIASES.get(path, path)


def view_label(path: str) -> str:
    key = canonical(path)
    for href, label in VIEWS:
        if href == key:
            return label
    return key


def nav_html(active: str) -> str:
    """The header: product + page name, then the view tabs with the current one marked."""
    active = canonical(active)
    links = []
    for href, label in VIEWS:
        if href == active:
            links.append(f'<a class="badge on" href="{href}" aria-current="page">{label}</a>')
        else:
            links.append(f'<a class="badge" href="{href}">{label}</a>')
    return (f'<h1>{PRODUCT} · {view_label(active)}</h1>'
            f'<nav class="views" aria-label="views">{"".join(links)}</nav>'
            f'<div class="strip" id="zgx-strip" aria-live="polite">reading /api/live…</div>')


def inject(html: str, active: str) -> str:
    """Put the shared frame into a page: stylesheet, header, status-strip script.

    Raises ValueError when a marker is absent. Serving that page anyway would give a reader
    a page with no navigation, no strip, or unstyled content -- a silent degradation is
    worse than a loud 500 that names the page.
    """
    missing = [m for m in (MARKER_HEAD, MARKER, MARKER_JS) if m not in html]
    if missing:
        raise ValueError("page is missing chrome marker(s): " + ", ".join(missing))
    html = html.replace(MARKER_HEAD, head_html(), 1)
    html = html.replace(MARKER, nav_html(active), 1)
    return html.replace(MARKER_JS, script_html(), 1)


def head_html() -> str:
    """The shared stylesheet, to be placed before a page's own <style> so page rules win."""
    return f"<style>{CHROME_CSS}</style>"


def script_html() -> str:
    """The shared status-strip script, to be placed at the end of the page body."""
    return CHROME_JS
