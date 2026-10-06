# ZGX / DGX Spark per-box monitor — the Prometheus exporter as a container.
#
# Stdlib only: no pip installs, so the image is just python + the two .py files.
# `nvidia-smi` is injected by the NVIDIA container runtime (CDI) at `docker run
# --gpus all` time, so it is deliberately NOT installed here.
# Host networking is required so the exporter's 127.0.0.1:18300 serving-health
# probe reaches the BOX's engine rather than the container's own loopback.
FROM python:3.12-slim

WORKDIR /monitor
COPY zgx_exporter.py corrected_metrics.py ./

ENV PYTHONUNBUFFERED=1 \
    TZ=Europe/Belgrade

EXPOSE 9400

HEALTHCHECK --interval=60s --timeout=10s --retries=3 --start-period=10s \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9400/health',timeout=4).status==200 else 1)"

CMD ["python", "/monitor/zgx_exporter.py"]
