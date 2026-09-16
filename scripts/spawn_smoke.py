#!/usr/bin/env python
"""Exercise bounded BATS and TBATS fit/forecast paths with explicit spawn."""

import warnings

import numpy as np

from tbats import BATS, TBATS


Y = 10.0 + np.sin(np.arange(32) * 2 * np.pi / 4) + 0.2 * np.cos(np.arange(32) * 2 * np.pi / 8)
COMMON_OPTIONS = {
    "use_box_cox": False,
    "use_trend": False,
    "use_damped_trend": False,
    "use_arma_errors": False,
    "seasonal_periods": [4],
    "multiprocessing_start_method": "spawn",
}


def fit_and_forecast(estimator_class, n_jobs):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        model = estimator_class(n_jobs=n_jobs, **COMMON_OPTIONS).fit(Y)
        forecast = model.forecast(steps=3)

    if caught_warnings:
        messages = ", ".join(str(warning.message) for warning in caught_warnings)
        raise AssertionError(f"{estimator_class.__name__} n_jobs={n_jobs} emitted warnings: {messages}")

    for name, values in (("fitted values", model.y_hat), ("forecast", forecast), ("AIC", [model.aic])):
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"{estimator_class.__name__} n_jobs={n_jobs} produced non-finite {name}")

    return model, forecast


def run_estimator(estimator_class):
    serial_model, serial_forecast = fit_and_forecast(estimator_class, n_jobs=1)
    parallel_model, parallel_forecast = fit_and_forecast(estimator_class, n_jobs=2)

    np.testing.assert_allclose(serial_model.y_hat, parallel_model.y_hat, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(serial_forecast, parallel_forecast, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(serial_model.aic, parallel_model.aic, rtol=1e-8, atol=1e-8)
    print(
        f"{estimator_class.__name__}: spawn n_jobs=1/2 agree; "
        f"warnings=0; aic={serial_model.aic:.12f}; forecast={np.round(serial_forecast, 8).tolist()}"
    )


def main():
    for estimator_class in (BATS, TBATS):
        run_estimator(estimator_class)


if __name__ == "__main__":
    main()
