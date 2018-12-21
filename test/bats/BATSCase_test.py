import numpy as np

from tbats.bats import Components, Case, Context


class TestBATSCase(object):

    def create_case(self, **components):
        return Case(Components(**components), Context())

    def test_fit_only_alpha(self):
        alpha = 0.7
        np.random.seed(333)
        T = 200

        l = 1
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + d
            l = l + alpha * d

        bats_case = self.create_case(use_trend=False)
        model = bats_case.fit(y)
        # assert estimated values are close to actual values
        assert np.isclose(alpha, model.params.alpha, atol=0.1)

    def test_fit_bats_case_with_trend(self):
        alpha = 0.8
        beta = 0.4
        np.random.seed(123)
        T = 400

        b = 0
        l = 1
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + b + d
            l = l + b + alpha * d
            b = b + beta * d

        bats_case = self.create_case(use_trend=True)
        model = bats_case.fit(y)
        # assert estimated values are close to actual values
        assert np.isclose(alpha, model.params.alpha, atol=0.1)
        assert np.isclose(beta, model.params.beta, atol=0.2)

    def test_fit_bats_case_with_damped_trend(self):
        alpha = 0.4
        beta = 0.6
        phi = 0.9
        np.random.seed(987)
        T = 500

        b = 0
        b_long = 0.0
        l = 1
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal(scale=1.0)
            y[t] = l + b + d
            l = l + b + alpha * d
            b = (1 - phi) * b_long + phi * b + beta * d

        bats_case = self.create_case(use_trend=True, use_damped_trend=True)
        model = bats_case.fit(y)
        # assert estimated values are close to actual values
        assert np.isclose(alpha, model.params.alpha, atol=0.1)
        assert np.isclose(beta, model.params.beta, atol=0.1)
        assert np.isclose(phi, model.params.phi, atol=0.1)

    def test_fit_with_boxcox(self):
        np.random.seed(49385)
        y = (np.random.uniform(size=100) + np.asarray(
            range(40, 140)) / 20) ** 2  # variance is increasing in those series

        bats_case = self.create_case(use_box_cox=True)
        model = bats_case.fit(y)

        assert np.isclose(0.4948, model.params.boxcox_lambda, atol=0.01)
        assert not np.allclose(model.resid, model.resid_boxcox)

    # TODO this test should pass
    # def test_fit_with_simple_seasonality(self):
    #     alpha = 0.7
    #     gamma = 0.4
    #     season_length = 4
    #     seasonal_starting_params = [-0.5, 0.5, -0.3, 0.3]  # sum is 0
    #     np.random.seed(32234)
    #     T = 300
    #
    #     l = 1
    #     y = [0] * T
    #     s = [0] * (T + season_length)
    #     s[0:season_length] = seasonal_starting_params
    #     for t in range(0, T):
    #         d = np.random.normal()
    #         s[t + season_length] = s[t] + gamma * d
    #         y[t] = l + s[t + season_length] + d
    #         l = l + alpha * d
    #
    #     model = BATSCase.fit_case(y, BATSComponents(seasonal_periods=[season_length]))
    #     # assert estimated values are close to actual values
    #     assert np.isclose(alpha, model.params.alpha, atol=0.1)
    #     assert np.isclose(gamma, model.params.gamma_params[0], atol=0.1)

    # TODO this test is not working as bats finds a better solution with different params than input ones :(
    # def test_fit_with_ar_errors(self):
    #     alpha = 0.8
    #     q1 = 0.5
    #     q2 = 0.3
    #     np.random.seed(222)
    #     T = 6000
    #
    #     l = 1
    #     e = 0
    #     e_prev_1 = 0
    #     e_prev_2 = 0
    #     d = 0
    #     y = [0] * T
    #     for t in range(0, T):
    #         y[t] = l + d
    #         l = l + alpha * d
    #
    #         d = q1 * e_prev_1 + q2 * e_prev_2 + e  # MA(2) model for errors
    #         e_prev_2 = e_prev_1
    #         e_prev_1 = e
    #         e = np.random.normal(scale=3.0)
    #
    #     case = BATSCase(use_arma_errors=True)
    #     model = case.fit(y)
    #     #assert estimated values are close to actual values
    #     assert np.isclose(alpha, model.params.alpha, atol=0.1)
