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

echo 'Local release preflight passed. Publishing is performed only by pushing a protected v<version> tag.'
echo 'First bump tbats.__version__, commit the release, ensure all CI jobs are green, then push its protected tag.'
