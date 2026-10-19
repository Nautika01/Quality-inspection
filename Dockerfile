# Offline-capable image: build once on a connected machine, ship the image file to the closed network.
#   docker build -t qinspect:0.3 .
#   docker pull eclipse-mosquitto:2 && docker save qinspect:0.3 eclipse-mosquitto:2 | gzip > qinspect-0.3-bundle.tar.gz
#   docker load < qinspect-0.3.tar.gz
#   docker run -p 8000:8000 -v qinspect-data:/data qinspect:0.3
# (a bind mount instead of the named volume must be writable by uid 1000: `chown -R 1000 /srv/qinspect`)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
WORKDIR /app

# dependencies come from the lock file (the versions the test suite ran against), the package itself on top
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

RUN useradd -m qinspect && mkdir -p /data && chown qinspect /data
USER qinspect
VOLUME ["/data"]
EXPOSE 8000 8501

ENV QINSPECT_DB=/data/qinspect.db
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["qinspect", "serve", "--db", "/data/qinspect.db", "--host", "0.0.0.0", "--port", "8000"]
