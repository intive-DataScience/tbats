#!/bin/bash

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo 'uv is required. Install uv, then rerun this script.' >&2
    exit 1
fi

uv sync --locked
uv run --locked python -m pytest test/
uv run --locked python scripts/spawn_smoke.py
rm -rf dist
uv build --no-sources
uvx --from twine==7.0.0 twine check dist/*
