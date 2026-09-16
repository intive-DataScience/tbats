#!/bin/bash

set -euo pipefail

if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
  echo 'Publishing requires a clean Git revision. Commit or remove all staged, unstaged, and untracked changes first.'
  exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "master" ]]; then
  echo 'Publish can be done only on master branch. Aborting.';
  exit 1;
fi

./prepare_package.sh

echo 'Local release preflight passed. This script never publishes.'
echo 'After bumping tbats.__version__ and confirming green CI, create and push a protected signed v<version> tag.'
echo 'Then publish a GitHub Release for that existing tag; the published Release triggers PyPI Trusted Publishing.'
