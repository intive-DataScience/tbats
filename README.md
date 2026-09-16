# BATS and TBATS time series forecasting

Package provides BATS and TBATS time series forecasting methods described in:

> De Livera, A.M., Hyndman, R.J., & Snyder, R. D. (2011), Forecasting time series with complex seasonal patterns using exponential smoothing, Journal of the American Statistical Association, 106(496), 1513-1527.


## Installation

From pypi:

```bash
python -m pip install tbats
```

Import via:

```python
from tbats import BATS, TBATS
```

## Minimal working example:

```python
from tbats import TBATS
import numpy as np

# required on windows for multi-processing,
# see https://docs.python.org/2/library/multiprocessing.html#windows
if __name__ == '__main__':
    np.random.seed(2342)
    t = np.array(range(0, 160))
    y = 5 * np.sin(t * 2 * np.pi / 7) + 2 * np.cos(t * 2 * np.pi / 30.5) + \
        ((t / 20) ** 1.5 + np.random.normal(size=160) * t / 50) + 10
    
    # Create estimator
    estimator = TBATS(seasonal_periods=[14, 30.5])
    
    # Fit model
    fitted_model = estimator.fit(y)
    
    # Forecast 14 steps ahead
    y_forecasted = fitted_model.forecast(steps=14)
    
    # Summarize fitted model
    print(fitted_model.summary())
```

Reading model details

```python
# Time series analysis
print(fitted_model.y_hat) # in sample prediction
print(fitted_model.resid) # in sample residuals
print(fitted_model.aic)

# Reading model parameters
print(fitted_model.params.alpha)
print(fitted_model.params.beta)
print(fitted_model.params.x0)
print(fitted_model.params.components.use_box_cox)
print(fitted_model.params.components.seasonal_harmonics)
```

See **examples** directory for more details.

## Troubleshooting

BATS and TBATS tries multitude of models under the hood and **may appear slow when fitting** to long time series. In order to speed it up you can start with constrained model search space. It is recommended to run it without Box-Cox transformation and ARMA errors modelling that are the slowest model elements:

```python
# Create estimator
estimator = TBATS(
    seasonal_periods=[14, 30.5],
    use_arma_errors=False,  # shall try only models without ARMA
    use_box_cox=False  # will not use Box-Cox
)
fitted_model = estimator.fit(y)
```

In some environment configurations parallel computation of models freezes. Reason for this is unclear yet. If **the process appears to be stuck** you can try running it on a single core:

```python
estimator = TBATS(
    seasonal_periods=[14, 30.5],
    n_jobs=1
)
fitted_model = estimator.fit(y)
```

## For Contributors

### Setup

Create and activate a virtual environment, then install the project and development tools:

```bash
python -m pip install -e '.[dev]'
```

`requirements.txt` and `requirements-dev.txt` are reviewed, fully pinned **macOS CPython 3.13** snapshots (Darwin arm64). They are not portable locks for Linux or for Python 3.10–3.12; those environments resolve compatible dependencies from project metadata. To reproduce the macOS CPython 3.13 development snapshot:

```bash
python -m pip install --upgrade -r requirements-bootstrap.txt
python -m pip install --no-deps -r requirements-dev.txt
python -m pip install --no-deps -e .
python -m pip check
```

`requirements-bootstrap.txt` pins the pip and setuptools versions used for this reproduction path. Regenerate the snapshots only on macOS CPython 3.13; `update_dependencies.sh` verifies both requirements before writing either file:

```bash
./update_dependencies.sh
```

### Testing

Run the non-R unit and integration suite:

```bash
python -m pytest test/
```

Run the bounded explicit-spawn smoke check for BATS and TBATS:

```bash
python scripts/spawn_smoke.py
```

R forecast package comparison tests are separate from normal development, CI, and release validation. They require R, the R `forecast` package, and the optional Python R extra:

```bash
python -m pip install '.[r]'
python -m pytest test_R/
```

### Release checks

Run the reviewed snapshot validation and build checks before a release:

```bash
./prepare_package.sh
python -m build
python -m twine check dist/*
```

`prepare_package.sh` and `publish_package.sh` install the bootstrap first, then run the non-R suite and explicit-spawn smoke check. Publishing is a separate manual action; it requires a clean Git revision and all CI jobs to be green. These instructions do not upload artifacts.

## Comparison to R implementation

Python implementation is meant to be as much as possible equivalent to R implementation in forecast package.

- BATS in R https://www.rdocumentation.org/packages/forecast/versions/8.4/topics/bats
- TBATS in R: https://www.rdocumentation.org/packages/forecast/versions/8.4/topics/tbats




