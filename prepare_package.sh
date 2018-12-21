#!/bin/bash

python -m pip install --user --upgrade setuptools wheel

python setup.py test || exit_on_error "Tests are not passing"
#python setup.py test_r || exit_on_error "R comparison tests are not passing"

pip-compile --output-file requirements.txt setup.py
pip freeze > requirements_stable.txt