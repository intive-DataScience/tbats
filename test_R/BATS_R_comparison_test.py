import numpy as np

from tbats import BATS
from .RComparisonBase import RComparisonBase


class TestBATSComparisonR(RComparisonBase):
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

        r_summary, r_model = self.r_bats(y, components)

        estimator = BATS(**components)
        py_model = estimator.fit(y)

        self.compare_model(r_summary, r_model, py_model)
        self.compare_forecast(r_model, py_model)

    def test_simple_seasonality(self):
        season_length = 5
        components = dict(
            use_arma_errors=False,
            use_trend=False,
            use_box_cox=False,
            seasonal_periods=[season_length],
        )

        np.random.seed(3244)
        gamma = 0.4
        seasonal_starting_params = [-0.5, 0.5, 0.8, -0.8, 0.0]  # sum is 0
        T = 50

        l = 1
        y = [0] * T
        s = [0] * (T + season_length)
        s[0:season_length] = seasonal_starting_params
        for t in range(0, T):
            d = np.random.normal()
            s[t + season_length] = s[t] + gamma * d
            y[t] = l + s[t + season_length] + d

        r_summary, r_model = self.r_bats(y, components)

        estimator = BATS(**components)
        py_model = estimator.fit(y)

        self.compare_model(r_summary, r_model, py_model)
        self.compare_forecast(r_model, py_model)

    def test_seasonality_with_subseasonality(self):
        season_length = 4

        components = dict(
            use_arma_errors=False,
            use_trend=False,
            use_box_cox=False,
            seasonal_periods=[2, season_length],
        )

        np.random.seed(3244)

        gamma = 0.6
        seasonal_starting_params = [-0.2, 0.2, -0.8, 0.8]  # sum is 0
        T = 50

        l = 1
        y = [0] * T
        s = [0] * (T + season_length)
        s[0:season_length] = seasonal_starting_params
        for t in range(0, T):
            d = np.random.normal()
            s[t + season_length] = s[t] + gamma * d
            y[t] = l + s[t + season_length] + d

        r_summary, r_model = self.r_bats(y, components)

        estimator = BATS(**components)
        py_model = estimator.fit(y)

        self.compare_model(r_summary, r_model, py_model)
        self.compare_forecast(r_model, py_model)

    def test_2_seasonalities_with_common_divisor(self):
        season_length_1 = 4
        season_length_2 = 6  # common divisor is 2

        components = dict(
            use_arma_errors=False,
            use_trend=False,
            use_box_cox=False,
            seasonal_periods=[season_length_1, season_length_2],
        )

        np.random.seed(3214)
        gamma1 = 0.6
        gamma2 = 0.5
        seasonal_starting_params_1 = [-0.2, 0.2, -0.8, 0.8]  # sum is 0
        seasonal_starting_params_2 = [-0.3, 0.2, 0.6, -0.6, 0.3, -0.2]  # sum is 0
        T = 300

        l = 1
        y = [0] * T
        s1 = [0] * (T + season_length_1)
        s1[0:season_length_1] = seasonal_starting_params_1
        s2 = [0] * (T + season_length_2)
        s2[0:season_length_2] = seasonal_starting_params_2
        for t in range(0, T):
            d = np.random.normal()
            s1[t + season_length_1] = s1[t] + gamma1 * d
            s2[t + season_length_2] = s2[t] + gamma2 * d
            y[t] = l + s1[t + season_length_1] + s1[t + season_length_1] + d

        r_summary, r_model = self.r_bats(y, components)

        estimator = BATS(**components)
        py_model = estimator.fit(y)

        self.compare_model(r_summary, r_model, py_model, atol_small=0.005, atol_for_series=0.005)
        self.compare_forecast(r_model, py_model, atol=0.005)

    def test_arma_errors(self):
        # Note: this test is very fragile to optimization function changes
        # as it is not easy to converge those series with ARMA errors.
        components = dict(
            use_arma_errors=True,
            use_trend=False,
            use_box_cox=False,
        )

        np.random.seed(3244)
        alpha = 0.9
        T = 100

        l = 1
        e = 0
        e_prev = 0
        e_prev_2 = 0
        y = [0] * T
        for t in range(0, T):
            e_prev_3 = e_prev_2
            e_prev_2 = e_prev
            e_prev = e
            e = np.random.normal()
            d = e + 0.1 * e_prev - 0.5 * e_prev_2 + 0.3 * e_prev_3  # AR(3) for errors
            y[t] = l + alpha * d

        r_summary, r_model = self.r_bats(y, components)

        estimator = BATS(**components)
        py_model = estimator.fit(y)

        assert py_model.params.components.arma_length() > 0  # there are ARMA components in the final model
        self.compare_model(r_summary, r_model, py_model, atol_small=0.01, atol_for_series=0.01)
        self.compare_forecast(r_model, py_model)

    def test_fit_boxcox(self):
        components = dict(
            use_arma_errors=False,
            use_trend=True,
            use_damped_trend=False,
            use_box_cox=True,
        )

        np.random.seed(49385)
        y = np.exp(
            np.random.uniform(size=100) + np.array(range(40, 140)) / 20
        )  # variance is increasing in those series

        r_summary, r_model = self.r_bats(y, components)

        estimator = BATS(**components)
        py_model = estimator.fit(y)

        self.compare_model(r_summary, r_model, py_model, atol_for_series=0.2)
        self.compare_forecast(r_model, py_model, atol=0.2)
