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

### Setup and locked development environment

Install [uv](https://docs.astral.sh/uv/) 0.12.3, then create the locked development environment:

```bash
uv sync --locked
```

The committed `uv.lock` is a universal development and CI lock for Python 3.10–3.13. It is not a consumer installation requirement; consumers install the package with pip or another standards-compliant installer. Update it deliberately after dependency changes:

```bash
uv lock
```

### Testing

Run the non-R unit and integration suite:

```bash
uv run --locked python -m pytest test/
```

Run the bounded explicit-spawn smoke check for BATS and TBATS:

```bash
uv run --locked python scripts/spawn_smoke.py
```

R forecast package comparison tests are separate from normal development, CI, and release validation. They require R, the R `forecast` package, and the optional Python R extra:

```bash
uv sync --locked --extra r
uv run --locked --extra r python -m pytest test_R/
```

If R packages live in a custom user library, set `R_LIBS_USER` for that command (for example, `R_LIBS_USER=/path/to/R/library uv run --locked --extra r python -m pytest test_R/`).

### Release checks

Run the reviewed snapshot validation and build checks before a release:

```bash
./prepare_package.sh
uv build --no-sources
uvx --from twine==7.0.0 twine check dist/*
```

`prepare_package.sh` runs the locked non-R suite, explicit-spawn smoke check, build, and metadata check. `publish_package.sh` is a local preflight only; it never uploads or creates tags.

To release a new version, bump `tbats.__version__`, commit it on `master`, ensure all CI jobs are green, and push the protected `v<version>` tag. The tag workflow validates the tag commit is on `master`, rebuilds a fresh `dist/`, and fails closed unless it contains exactly one matching wheel and sdist. It validates both embedded metadata files and records SHA-256 hashes before installing and smoking the exact wheel externally. The later upload is bound to those two validated paths; the publish job downloads that exact artifact, validates it again, recomputes and compares both hashes, and only then publishes through PyPI Trusted Publishing. Existing version 1.1.3 cannot be republished.

One-time release administration: configure the PyPI Trusted Publisher with owner `intive-DataScience`, repository `tbats`, workflow `publish.yml`, and environment `pypi`. Protect the GitHub `pypi` environment and `v*` tags. No PyPI token secret is used.

## Comparison to R implementation

Python implementation is meant to be as much as possible equivalent to R implementation in forecast package.

- BATS in R https://www.rdocumentation.org/packages/forecast/versions/8.4/topics/bats
- TBATS in R: https://www.rdocumentation.org/packages/forecast/versions/8.4/topics/tbats

