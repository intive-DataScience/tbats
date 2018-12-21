#!/bin/bash

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "master" ]]; then
  echo 'Publish can be done only on master branch. Aborting.';
  exit 1;
fi

python setup.py register sdist bdist_wheel upload