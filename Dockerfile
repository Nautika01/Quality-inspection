# Offline-capable image: build once on a connected machine, ship the image file to the closed network.
#   docker build -t qinspect:0.2 .
#   docker save qinspect:0.2 | gzip > qinspect-0.2.tar.gz      # move this file inside
#   docker load < qinspect-0.2.tar.gz
#   docker run -p 8000:8000 -v /data:/data -e QINSPECT_DB=/data/qinspect.db qinspect:0.2
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[server]"

RUN useradd -m qinspect && mkdir -p /data && chown qinspect /data
USER qinspect
VOLUME ["/data"]
EXPOSE 8000 8501

ENV QINSPECT_DB=/data/qinspect.db
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["qinspect", "serve", "--db", "/data/qinspect.db", "--host", "0.0.0.0", "--port", "8000"]
