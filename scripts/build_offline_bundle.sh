#!/usr/bin/env bash
# Build a wheelhouse so the package installs on a machine with no internet (closed network).
#   ./scripts/build_offline_bundle.sh            -> dist/offline/
#   on the target:  pip install --no-index --find-links dist/offline -r requirements-lock.txt qinspect
# Versions are pinned by requirements-lock.txt; regenerate it (pip freeze in a clean venv + pytest) when upgrading.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=dist/offline
rm -rf "$OUT" && mkdir -p "$OUT"
pip wheel --wheel-dir "$OUT" -r requirements-lock.txt .
sha256sum "$OUT"/*.whl > "$OUT/SHA256SUMS"
cp requirements-lock.txt "$OUT/"
echo "wheelhouse at $OUT ($(ls "$OUT"/*.whl | wc -l) wheels); verify with: sha256sum -c $OUT/SHA256SUMS"
