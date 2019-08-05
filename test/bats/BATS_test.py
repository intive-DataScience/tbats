import pytest
import numpy as np

import tbats.error as error
from tbats import BATS


class TestBATS(object):

    def test_input_validation(self):
        estimator = BATS()
        with pytest.raises(error.InputArgsException):
            estimator.fit([])
        with pytest.raises(error.InputArgsException):
            estimator.fit('string')

    def test_seasonal_periods_input_validation(self):
        with pytest.warns(error.InputArgsWarning):
            BATS(seasonal_periods=[1, 3])
        with pytest.warns(error.InputArgsWarning):
            BATS(seasonal_periods=[0, 3])

    def test_constant_model(self):
        y = [2.9] * 20
        estimator = BATS()
        model = estimator.fit(y)
        assert np.allclose([0.0] * len(y), model.resid)
        assert np.allclose(y, model.y_hat)
        assert np.allclose([2.9] * 5, model.forecast(steps=5))

    def test_fit_only_alpha(self):
        alpha = 0.8
        np.random.seed(333)
        T = 200

        l = 1
        y = [0] * T
        for t in range(0, T):
            d = np.random.normal()
            y[t] = l + d
            l = l + alpha * d

        estimator = BATS()
        model = estimator.fit(y)
        assert np.isclose(alpha, model.params.alpha, atol=0.1)
