#!/bin/bash

set -euo pipefail

PYTHON="${PYTHON:-python}"

"$PYTHON" -m pip install --upgrade -r requirements-bootstrap.txt
"$PYTHON" -m pip install --upgrade --no-deps -r requirements-dev.txt
"$PYTHON" -m pip install --no-deps -e .
"$PYTHON" -m pip check
"$PYTHON" -m pytest test/
"$PYTHON" scripts/spawn_smoke.py
"$PYTHON" -m build
