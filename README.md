# runboard-lcp

The ZGX run board: the read-only telemetry pages for the DGX Spark (`192.168.1.107`), the
dispatch console that queues engine switches, evals and downloads, and the dispatcher that
executes them out-of-process.

Three views and one JSON surface, all served by a single stdlib HTTP server on port `18400`:

| path | what it is |
|---|---|
| `/` | now, and only now — engine load (every request, run or not: the decode pair — single stream / aggregate — prefill, DECODING lanes against the engine's own cap, last-generation recency), refreshed once a second, and per-Spark telemetry. Fetches `/api/live` and nothing else. |
| `/history` | every banked run, ranked inside its own kit — plus the load-soak archive with its date, its model and its thermal envelope |
| `/console` | switch engine, queue an eval, search HuggingFace and download a pack |
| `/api/state` `/api/history` `/api/models` `/api/dispatch` `/api/live` | the JSON behind those pages |

**The split is a rule, not a layout choice.** `/` is the live page and carries no archived number;
`/history` is the archive page and carries no live one. Every archive number states the date and
the model that produced it, at the point of display. This is not cosmetic: `/` used to fetch
`/api/state` (the soak archive, 67 KB) every 3 s and render it as a Throughput card reading
`0.00 tok/s` beside an engine that was serving — a retired model's zero, four seconds' walk from a
live reading, reads as *the box is idle*. Measured cost of that: 27 KB/s per open tab, 1.95 GB/day.
The archive is still complete; it is on the page that says what it is (`tests/test_page_badges.py`
pins both halves of this).

The serving engine itself is a separate concern: exllamav3 on `:18300` behind
`/root/serve.sh` (`cruz` \| `exl3` \| `vllm` \| `status`). This repo reads and drives that
engine; it is not the engine.

## Layout

```
console_core.py       shared core: the queue, the presets, path/token helpers, HF queries.
                      Imported by BOTH the web server and the dispatcher, so the two can
                      never disagree about what a job is.
registry.py           builds models.json -- what the box can serve, what is serving now,
                      what each pack measured, and its size on disk.
live_server.py        the read-only HTTP server (:18400). Pages + JSON. Executes nothing.
dispatcher.py         the executor. Runs as load-dispatch.service on a 15s timer.
history_collector.py  folds run artifacts into run/history.json for /history.
zgx_exporter.py       Prometheus exporter (:9400) for the Grafana dashboard.
engine_metrics.py     what the SERVING ENGINE is doing right now: the decode PAIR (single stream =
                      generated tokens per second of decode time; aggregate = generated tokens per
                      second of wall clock for the whole engine), prefill tok/s, TTFT, KV occupancy,
                      prefix-cache hit rate, MTP acceptance, DECODING lanes against the engine's
                      own `--parallel` cap -- read off :18300 and
                      Prometheus (job zgx-vllm), so the page answers "what is the box doing" with
                      no eval running. Metric names are aliased per engine family
                      (FAMILY_ALIASES / FAMILY_EXPR); a name with no alias dashes silently, which
                      is why every expression is checked against the engine's captured inventory.
corrected_metrics.py  the one definition of aggregate throughput (tokens / wall clock).
load_soak.py          the load-soak harness that produced the archive on `/history`. Source-of-
                      record only; not deployed (a soak is an operator action, not a request).
static/               the three pages (deployed flat into /root/load).
systemd/              the units, as installed on the box.
monitor/              one monitoring container per Spark (see monitor/README.md). Each GB10
                      box serves one engine in parallel (TP=2), so each carries its own
                      exporter on :9400; the board reads the pair, not only this box.
scripts/serve.sh      the engine switcher, as installed at /root/serve.sh.
scripts/deploy.sh     ship this checkout to the box and smoke the endpoints.
tests/                unit tests. See "Tests" below.
```

## Design rules

These are not style preferences; each one is the fix for something that actually went wrong.

1. **The web process never executes anything.** `/api/dispatch` authenticates a request and
   then writes a job file into an atomic-rename queue. It does not run it. Execution lives in
   `dispatcher.py`, a separate systemd unit. A web process that can switch the serving engine
   is a web process that can take the box down from a stray request.
2. **Writes need the bearer token** (`/root/load/dispatch.token`, root-only). Reads are open;
   the pages have to be loadable by a browser without a header.
3. **One lock, held across a whole job.** Engine switches, evals and downloads all take the
   same `flock`, so two jobs can never contend for `:18300`.
4. **The baseline is restored after every eval.** `BASELINE = "cruz"` in `console_core.py`.
   A job may name any engine; when it finishes, the box goes back to the baseline. An explicit
   `serve` job is the one exception -- that is the operator naming the desired state -- and
   when it leaves the box off baseline, `status.json` records `off_baseline` and the console
   shows it rather than hiding the drift.
5. **`status.json` is a receipt.** It preserves the last job's record across idle ticks so the
   page can answer "is the box back on the baseline?" after the job is long gone.
6. **A missing measurement is a dash, never a zero.** Never blend decode-only throughput with
   end-to-end, and never rank across kits. Unknown disk size reports as unknown. A rate that is
   `null` means "no such traffic in that interval" (no request FINISHED, say) -- it is not 0.
   `/api/live -> engine` is the surface where that distinction is load-bearing: a dash there and
   the engine is idle, a 0.0 there and someone is being told the box is not working when it is.
7. **The board answers about the ENGINE, not only about runs.** A run card plus a soak archive
   left every non-eval use of the box (agent traffic, a manual probe) invisible: on 2026-09-28 a
   live generation at 20 tok/s with nothing queued rendered as the previous day's soak. Anything
   the box does has to be visible without a run being queued.
8. **Do not guess a key name.** The dispatcher's first summary reader guessed `auto_graded`
   and reported all-null against an artifact that was fine; the HF search date field is
   `createdAt`, not `lastModified`. Read the names the source actually uses.
9. **A page shows one time horizon.** `/` is live and `/history` is archive; neither carries the
   other's numbers. The rule follows from rule 6 rather than from taste: an archived `0.00 tok/s`
   on the live page is the same lie as a zero for an unreachable box, because the reader has no
   way to know it is a different model on a different day. Where an archived number is shown at
   all it carries its date and its model *in the value's own label* -- `aggregate tok/s (60 s
   rolling, 2026-09-24 13:05)` -- so its age is structural and cannot be forgotten. The board
   kept a 12-day-old soak on `/` until 2026-10-06 and it read as a current reading.
10. **One actionable signal per page.** On `/` only temperature is coloured, at 82/85 °C. A page
   where four things compete for attention is a page where none of them holds it, and the one
   that someone would actually act on is the one that gets lost.
11. **A per-stream rate is meaningless without the regime it was measured in.** "decode tok/s" can
   mean *one stream with the box to itself* or *one stream while seven others share it*, and those
   differ by an order of magnitude -- printing either without saying which is a claim the number
   cannot support. The engine card therefore carries **two** numbers and only two (operator,
   2026-10-06: "can we just have 19.11 tps decode single stream, 80 tps decode aggregate. Simplify
   it."): `single stream` = generated tokens per second of *decode time* (the per-stream speed, which
   does not sag when lanes share) and `aggregate` = generated tokens per second of *wall clock* for
   the whole engine. The lane pair above (`in use` against the engine's own cap) says which regime
   you are looking at.
   They are **two independent measurements over two different counter families**, so do not divide
   one by the other and read the lane count: measured 2026-10-06 that quotient came out at 6.6 while
   four lanes were held and 5.9 while eight were held, because the numerator counts only *finished*
   requests while the aggregate covers the whole engine. Two answers, not a ratio.
   An earlier revision of this card that same day carried a third reading ("one stream, alone") and a
   derived per-stream rate. Both measured the same quantity as the pair and the operator asked for
   them to go, so they now sit in the collapsed diagnostics row rather than being deleted outright.
   **The pair is not a ratio, but neither is the concurrency a gauge you can guess at.** Two more
   faults were fixed on 2026-10-06, both found by the operator reading the live card and saying
   *"something's incorrect here"* -- and both were the card's, not the engine's:

   * **`requests_running` is not a stream count.** The engine's own help text: *"Requests in prefill
     or decode."* Sampling every 5 s for 120 s confirms
     `requests_running == streams{state="decoding"} + streams{state="filling"}` exactly (means 6.21
     == 5.75 + 0.46), so it counts requests parked in **prefill**, which emit no decode tokens. That
     is how "8 concurrent streams" sits beside an aggregate *below* the single-stream rate: 8
     requests held, well under one lane of real decode work. The card labels it honestly now and
     shows `decoding` (a per-state gauge) as the concurrency. Reading one state out of a
     multi-series metric needs a **label-qualified key**, which `parse_engine` keeps alongside the
     bare name -- summing the three states answers a different question.
   * **A per-stream rate divided out of a gauge is not a measurement.** `stream_tps`
     (`output_tps / running`) is deleted. It divided a two-minute *time average* by an
     *instantaneous snapshot*, describing no single moment: measured **1.21** tok/s against a true
     per-lane **13.5**, and **48.4** against **16.1** an hour earlier -- arbitrary, not biased.
     `decode_tps` (`tokens / decode seconds`) already is the per-stream rate.

   Ground truth for the pair, measured over 120 s: **4.63 lanes** of decode work produced
   **78.4 tok/s = 16.9 tok/s per lane**, with `decode_tps` reading 13--16.5. And note *why* the
   aggregate can read low while the lanes are fast: it is a wall-clock average, so it sags toward
   zero whenever the box pauses between requests -- 8 s windows were caught with **6 streams held
   and zero tokens produced**. A low aggregate is idle time, not slow decoding.

   Corollary, learned on 2026-10-06: a metric name missing from `FAMILY_ALIASES` raises nothing --
   the expression simply matches nothing and the KPI dashes forever, which on this card reads as
   "no decode happened" (`decode_tps` did exactly that against the TensorFold engine).
   `test_every_name_the_card_queries_is_a_name_this_engine_publishes` is the guard: every name in
   every expression must appear in the engine's own captured inventory.
12. **Know the flush interval before you trust a rate off a counter.** The TensorFold engine does
   not publish a continuously-updated counter: both `generation_tokens_total` and
   `decode_seconds_total` advance in **lockstep lumps every ~8 s** (median of 11 gaps measured over
   92 s: 16/8/6/8/10/8/2/8/6/6/6 s, the same list for both). A 2 m `rate()` therefore spans ~15
   flushes, which is enough -- but read the consequence before drawing conclusions from a single
   sample: the 1.7x spread observed on `output_tps` across 13 samples (102-170 tok/s against ~95
   over 30 m) is **real load variation, not instrument noise**. The one number that stays stable
   under lumping is a *ratio of two counters that lump together*, which is precisely why
   `decode_tps` (13-17 across the same window) can be trusted while its numerator alone cannot.
   Caution on the other side: `decode_seconds_total` integrates to as much as **10.5 lanes at 2 m
   against an 8-lane cap**, so that denominator is good to roughly **+/-30 %**, not exact.
13. **Refresh as fast as the operator watches, and pay for it at the socket.** `/` polls once a
   second (operator, 2026-10-06: "the entire engine load must be live and refreshed once per
   second"). The next poll is scheduled from the END of the previous one rather than handed to
   `setInterval`: this page is served by the box that is running the engine, so a fixed interval
   would fire again while a slow response was still in flight and pile requests onto the engine.
   Every response of 1 KB or more is gzipped when the client asks for it (~4:1 on the live
   document), which is what makes three times the refresh rate cost *less* bandwidth than the 3 s
   poll it replaced: 6.9 -> 4.4 KiB/s per open tab, and the two pollers the page runs (1 s card,
   10 s chrome strip) are accounted for in that figure.

## Running it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt

# the modules are location-independent; the defaults are the production paths
python3 -c "import console_core as C; print(C.LOAD, C.RUNS_DIR)"
```

Every root is overridable by environment variable, which is what makes the tests hermetic:

| variable | default |
|---|---|
| `RUNBOARD_LOAD` | `/root/load` |
| `RUNBOARD_STATIC` | `$RUNBOARD_LOAD` |
| `RUNBOARD_RUNS_DIR` | `/root/exl3-bench/runs` |
| `RUNBOARD_BENCH` | `/root/exl3-bench` |
| `RUNBOARD_HF_CACHE` | `/root/.cache/huggingface/hub` |
| `RUNBOARD_SERVE` | `/root/serve.sh` |
| `RUNBOARD_RUNNER_LITE` | `/root/q200_lite.py` |
| `RUNBOARD_RUNNER_FROZEN` | `.../q200v2/scripts/run_quality_set.py` |

## Tests

```bash
pip install -r requirements-dev.txt
pytest                      # addopts: --cov --cov-report=term-missing
```

`tests/conftest.py` points every `RUNBOARD_*` variable at a throwaway sandbox before the
modules are imported and fails any test that reaches a production path. Tests must not need
the network, a GPU, a listening port, systemd, or the real `:18300`.

Coverage target: **100%** of the service modules (`console_core`, `registry`, `dispatcher`,
`live_server`, `history_collector`, `zgx_exporter`, `corrected_metrics`, `live_metrics`,
`engine_metrics`), enforced by `fail_under = 100`.

**Deliberately out of the coverage target**, and stated rather than quietly omitted:

- `load_soak.py` -- a long-running load harness that needs a live endpoint and a GPU; its
  correctness is established by its own smoke run and its negative control, not by unit tests.
- `scripts/serve.sh` and the systemd units -- shell/config, exercised by the health-gated
  switcher and `systemctl` at deploy time.

## Deploying

```bash
bash scripts/deploy.sh          # backup, ship, compile-gate, restart, smoke the endpoints
```

The deployment is flat (`/root/load/*.py`); the checkout keeps pages under `static/`. Nothing
in the deploy path touches the serving engine -- engine changes go through `serve.sh` or the
dispatcher, never through a deploy.
