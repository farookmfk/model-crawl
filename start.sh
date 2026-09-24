#!/usr/bin/env sh
# Linux/macOS launcher: creates the virtualenv on first run, then starts Model Crawl.
#   MODEL_CRAWL_NO_BROWSER=1  don't try to open a browser (headless servers)
#   PYTHON=python3.12        pick the interpreter used to create the venv (3.10+)
set -e
cd "$(dirname "$0")"

if [ ! -f .venv/.installed ]; then
    "${PYTHON:-python3}" -m venv .venv || {
        echo "Could not create a virtualenv. On Debian/Ubuntu: sudo apt install python3-venv" >&2
        exit 1
    }
    .venv/bin/python -m pip install -q -r requirements.txt
    touch .venv/.installed
fi

URL="http://127.0.0.1:${MODEL_CRAWL_PORT:-8765}"
if [ -z "$MODEL_CRAWL_NO_BROWSER" ]; then
    for opener in xdg-open open; do
        if command -v "$opener" >/dev/null 2>&1; then
            (sleep 2 && "$opener" "$URL" >/dev/null 2>&1) &
            break
        fi
    done
fi

exec .venv/bin/python server.py
