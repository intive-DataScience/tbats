#!/bin/bash

python -m pip install --upgrade setuptools wheel
python -m pip install --upgrade twine

python setup.py test || exit_on_error "Tests are not passing"
python setup.py test_r || exit_on_error "R comparison tests are not passing"

pip-compile --output-file requirements.txt setup.py
pip freeze > requirements_stable.txt