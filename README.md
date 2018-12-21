# BATS and TBATS time series forecasting

**This is an early ALPHA version !!!**

Package provides BATS and TBATS time series forecasting methods:

> De Livera, A.M., Hyndman, R.J., & Snyder, R. D. (2011), Forecasting time series with complex seasonal patterns using exponential smoothing, Journal of the American Statistical Association, 106(496), 1513-1527.


## Installation

From pypi:

```bash
pip install tbats
```

Import via:

```python
from tbats import BATS, TBATS
```

## Minimal working example:

```python
from tbats import TBATS
import numpy as np

np.random.seed(2342)
t = np.array(range(0, 160))
y = 5 * np.sin(t * 2 * np.pi / 14) + \
    ((t / 20) ** 1.5 + np.random.normal(size=160) * t / 50) + 10

# Create estimator
estimator = TBATS(seasonal_periods=[14])

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
print(fitted_model.aic())

# Reading model parameters
print(fitted_model.params.alpha)
print(fitted_model.params.beta)
print(fitted_model.params.x0)
print(fitted_model.params.components.use_box_cox)
print(fitted_model.params.components.seasonal_harmonics)

```

## For Contributors

Building package:

```bash
pip install -e .[dev]
```

Unit and integration tests:

```bash
python setup.py test
```

R forecast package comparison tests. Those DO NOT RUN with default test command, you need R forecast package installed:
```bash
python setup.py test_r
```



## Comparison to R implementation

Python implementation is meant to be as much as possible equivalent to R implementation in forecast package but will not provide exactly the same results as R package.

- BATS in R https://www.rdocumentation.org/packages/forecast/versions/8.4/topics/bats
- TBATS in R: https://www.rdocumentation.org/packages/forecast/versions/8.4/topics/tbats






