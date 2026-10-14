#!/usr/bin/env bash
# Build a wheelhouse so the package installs on a machine with no internet (closed network).
#   ./scripts/build_offline_bundle.sh            -> dist/offline/
#   on the target:  pip install --no-index --find-links dist/offline qinspect[server]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=dist/offline
rm -rf "$OUT" && mkdir -p "$OUT"
pip wheel --wheel-dir "$OUT" ".[server]"
sha256sum "$OUT"/*.whl > "$OUT/SHA256SUMS"
echo "wheelhouse at $OUT ($(ls "$OUT"/*.whl | wc -l) wheels); verify with: sha256sum -c $OUT/SHA256SUMS"
