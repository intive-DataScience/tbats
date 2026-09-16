#!/bin/bash

set -euo pipefail

# To test:
# ./publish_package.sh
# To production:
# ./publish_package.sh PRODUCTION

if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
  echo 'Publishing requires a clean Git revision. Commit or remove all staged, unstaged, and untracked changes first.'
  exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "master" ]]; then
  echo 'Publish can be done only on master branch. Aborting.';
  exit 1;
fi

PYTHON="${PYTHON:-python}"

"$PYTHON" -m pip install --upgrade -r requirements-bootstrap.txt
"$PYTHON" -m pip install --upgrade --no-deps -r requirements-dev.txt
"$PYTHON" -m pip install --no-deps -e .
"$PYTHON" -m pip check
"$PYTHON" -m pytest test/
"$PYTHON" scripts/spawn_smoke.py
rm -rf dist
"$PYTHON" -m build
"$PYTHON" -m twine check dist/*

if [ "${1:-}" == "PRODUCTION" ]; then
    "$PYTHON" -m twine upload dist/*
else
    "$PYTHON" -m twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*
fi
