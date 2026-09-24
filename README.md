# runboard-lcp

The ZGX run board: the read-only telemetry pages for the DGX Spark (`192.168.1.107`), the
dispatch console that queues engine switches, evals and downloads, and the dispatcher that
executes them out-of-process.

Three views and one JSON surface, all served by a single stdlib HTTP server on port `18400`:

| path | what it is |
|---|---|
| `/` | now — live load, per-shape throughput, soak summary |
| `/history` | every banked run, ranked inside its own kit |
| `/console` | switch engine, queue an eval, search HuggingFace and download a pack |
| `/api/state` `/api/history` `/api/models` `/api/dispatch` | the JSON behind those pages |

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
corrected_metrics.py  the one definition of aggregate throughput (tokens / wall clock).
load_soak.py          the load-soak harness that produced the numbers on `/`.
static/               the three pages (deployed flat into /root/load).
systemd/              the units, as installed on the box.
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
   end-to-end, and never rank across kits. Unknown disk size reports as unknown.
7. **Do not guess a key name.** The dispatcher's first summary reader guessed `auto_graded`
   and reported all-null against an artifact that was fine; the HF search date field is
   `createdAt`, not `lastModified`. Read the names the source actually uses.

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
`live_server`, `history_collector`, `zgx_exporter`, `corrected_metrics`), enforced by
`fail_under = 100`.

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
