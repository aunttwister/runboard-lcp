# monitor/ — one monitoring container per Spark

The ZGX / DGX Spark machine monitor, containerised so **each Spark carries its own**
monitoring container rather than a single hand-installed systemd unit on one box.

| Host | Role | Monitor |
|---|---|---|
| `192.168.1.107` | DGX Spark (HP ZGX G1n), TP rank 0 · head, load harness | `spark-monitor` container, `:9400` |
| `192.168.1.108` | MSI EdgeXpert, TP rank 1 · worker | `spark-monitor` container, `:9400` |

Both boxes serve **one** engine in parallel (TP=2, ~97 GB resident each), so each is half
of the machine. The board reads the PAIR (`live_metrics.DEFAULT_BOXES`), not only the box
it happens to run on: a one-box view reports half the machine as though it were the whole,
and the unseen half is exactly where a thermal event hides. Before 2026-10-06 only `.107`
had an exporter at all — the EdgeXpert, i.e. half the production pair, was thermally
invisible, and Prometheus scraped only `.107`.

Prometheus job `zgx-export` scrapes **both** on `:9400` (1 y retention), so Grafana and the
runboard see the whole pair.

## Why the container must run ON the box

Every metric here is sampled locally — `nvidia-smi` for GPU power/temp/clock/util, and
`/proc/meminfo` for memory and swap. A container elsewhere cannot see them, so this is the
one thing a remote poller cannot do for you.

It runs `network_mode: host` for two reasons: it must bind the box's `:9400`, and its
`zgx_serving_up` probe must reach `127.0.0.1:18300` — the *box's* engine, not the
container's loopback.

`nvidia-smi` is **not** in the image. The NVIDIA container runtime (CDI) injects it at
`docker run --gpus all` time, so the image stays stdlib-only Python.

## Deploy

```bash
bash monitor/deploy.sh root@192.168.1.107     # head
bash monitor/deploy.sh root@192.168.1.108     # worker
```

The script copies the two modules from the **checkout root** (not from `monitor/`), so the
container always ships the same exporter the rest of the repo tests, then rebuilds and
recreates the container. Idempotent.

### `.107` — the one manual step

`.107` used to run a hand-installed host-side unit, `zgx-exporter.service`, on the same
`:9400`. It must be stopped and **disabled** before the container can bind the port. This
is deliberately not automated — it changes what runs on a production box — and it is also
the whole rollback:

```bash
# switch to the container (once)
ssh root@192.168.1.107 'systemctl disable --now zgx-exporter.service'
bash monitor/deploy.sh root@192.168.1.107

# roll back to the unit
ssh root@192.168.1.107 'docker rm -f spark-monitor; systemctl enable --now zgx-exporter.service'
```

## Verify

```bash
curl -s http://192.168.1.107:9400/metrics | grep -c '^zgx_'   # ~50 series
curl -s http://192.168.1.108:9400/metrics | grep -c '^zgx_'
curl -s localhost:18400/api/live | python3 -m json.tool | grep -A3 '"boxes"'
```

`/api/live`'s `boxes` array must hold **two** entries with `"ok": true`. A box that does not
answer reports `"ok": false` and **no values** — the page then draws a dash, never a zero,
because 0 °C on an unreachable Spark reads exactly like a cold one.

## Alerting

The thermal alert is not here. It is the `spark-thermal-watchdog` cron, which reads both
exporters and emails (and rings) on a sustained breach: ≥82 °C for 10 min, ≥85 °C (the
published shutdown zone) immediately, with a "blind" alert if a box is unreachable for
15 min. This directory only *exposes* the numbers.
