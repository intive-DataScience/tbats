import numpy as np

from tbats import TBATS
from test_R.RComparisonBase import RComparisonBase


class TestTBATSComparisonR(RComparisonBase):
    """
    In order to run those tests you need installed R and it's forecast package
    """

    def test_damped_trend(self):
        components = dict(
            use_arma_errors=False,
            use_trend=True,
            use_damped_trend=True,
            use_box_cox=False
        )

        alpha = 0.4
        beta = 0.6
        phi = 0.9
        np.random.seed(987)
        T = 100

        b = 0
        b_long = 0.0
        l = 1
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal(scale=1.0)
            y[t] = l + b + d
            l = l + b + alpha * d
            b = (1 - phi) * b_long + phi * b + beta * d

        r_summary, r_model = self.r_tbats(y, components)

        estimator = TBATS(**components)
        py_model = estimator.fit(y)

        self.compare_model(r_summary, r_model, py_model)
        self.compare_forecast(r_model, py_model)

    def test_trend_and_seasonal(self):
        np.random.seed(234234)
        T = 35
        steps = 5
        alpha = 0.1
        period_length = 6
        y = [0] * T
        b = b0 = 2.1
        l = l0 = 1.2
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + b + d + 2 * np.sin(2 * np.pi * t / period_length)
            l = l + b + alpha * d

        components = dict(
            use_arma_errors=False,
            use_trend=True,
            use_damped_trend=False,
            use_box_cox=False,
            seasonal_periods=[period_length],
        )

        y_for_train = y[:(T - steps)]
        y_to_forecast = y[(T - steps):]

        r_summary, r_model = self.r_tbats(y_for_train, components)

        estimator = TBATS(**components)
        py_model = estimator.fit(y_for_train)

        self.assert_py_model_is_not_worse(y_for_train, r_summary, r_model, py_model)
        self.assert_forecast_is_not_worse(y_to_forecast, r_model, py_model)

    def test_trend_and_two_seasonalities(self):
        np.random.seed(234234)
        T = 35
        steps = 5
        alpha = 0.1
        period_1_length = 3
        period_2_length = 7
        y = [0] * T
        b = b0 = 2.1
        l = l0 = 1.2
        for t in range(0, T):
            d = np.random.normal()
            s1 = 2 * np.cos(2 * np.pi * t / period_1_length)
            s2 = 3 * np.sin(2 * np.pi * t / period_2_length)
            y[t] = l + b + s1 + s2 + d
            l = l + b + alpha * d

        components = dict(
            use_arma_errors=False,
            use_trend=True,
            use_damped_trend=False,
            use_box_cox=False,
            seasonal_periods=[period_1_length, period_2_length],
        )

        y_for_train = y[:(T - steps)]
        y_to_forecast = y[(T - steps):]

        r_summary, r_model = self.r_tbats(y_for_train, components)

        estimator = TBATS(**components)
        py_model = estimator.fit(y_for_train)

        self.assert_py_model_is_not_worse(y_for_train, r_summary, r_model, py_model)
        self.assert_forecast_is_not_worse(y_to_forecast, r_model, py_model)

    def test_long_seasonality(self):
        np.random.seed(5434)
        T = 300
        steps = 5
        alpha = 0.1
        period_1_length = 7
        period_2_length = 30.5
        y = [0] * T
        b = b0 = 2.1
        l = l0 = 1.2
        for t in range(0, T):
            d = np.random.normal()
            s1 = 2 * np.cos(2 * np.pi * t / period_1_length)
            s2 = 3 * np.sin(2 * np.pi * t / period_2_length)
            y[t] = l + b + s1 + s2 + d
            l = l + b + alpha * d

        components = dict(
            use_arma_errors=False,
            use_trend=True,
            use_damped_trend=False,
            use_box_cox=False,
            seasonal_periods=[period_1_length, period_2_length],
        )

        y_for_train = y[:(T - steps)]

        y_to_forecast = y[(T - steps):]

        r_summary, r_model = self.r_tbats(y_for_train, components)

        estimator = TBATS(n_jobs=1, **components)
        py_model = estimator.fit(y_for_train)

        self.assert_py_model_is_not_worse(y_for_train, r_summary, r_model, py_model)
        self.assert_forecast_is_not_worse(y_to_forecast, r_model, py_model)