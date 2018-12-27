#!/bin/bash

# To test:
# ./publish_package.sh
# To production:
# ./publish_package.sh PRODUCTION

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "master" ]]; then
  echo 'Publish can be done only on master branch. Aborting.';
  exit 1;
fi

python setup.py sdist bdist_wheel

if [ "$1" == "PRODUCTION" ]; then
    twine upload dist/*
else
    twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*
fi