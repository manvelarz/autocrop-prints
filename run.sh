#!/usr/bin/env bash
# Run autocrop-prints with its own venv, from any directory.
#
#   ./run.sh /path/to/scans                 -> writes /path/to/scans/cropped/
#   ./run.sh /path/to/scans /path/to/out    -> explicit output folder
#   ./run.sh /path/to/scans --dry-run       -> only report + contact sheet
#
# Anything after the first two arguments is passed straight to autocrop_prints.py,
# so --margin, --min-area, --max-dim, --ext all work.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "first run: setting up the venv (one time)"
  cd "$HERE"
  uv venv .venv --python 3.12 -q
  uv pip install -p .venv/bin/python opencv-python-headless numpy -q
fi

if [ $# -lt 1 ]; then
  sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 1
fi

IN="$1"; shift
if [ $# -gt 0 ] && [ "${1:0:1}" != "-" ]; then
  OUT="$1"; shift
  exec "$PY" "$HERE/autocrop_prints.py" "$IN" -o "$OUT" "$@"
fi
exec "$PY" "$HERE/autocrop_prints.py" "$IN" "$@"
