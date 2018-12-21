import numpy as np
from sklearn.linear_model import LinearRegression

from . import Components


class SeedFinder(object):

    def __init__(self, components):
        """
        :param Components components:
        :return:
        """
        self.components = components

    def to_matrix_for_linear_regression(self, w_tilda):
        # abstract method
        raise NotImplementedError()

    def from_linear_regression_coefs_to_x0(self, linear_regression_coefs):
        # abstract method
        raise NotImplementedError()

    def find(self, w_tilda, residuals):
        w_for_lr = self.to_matrix_for_linear_regression(w_tilda)

        linear_regression = LinearRegression(fit_intercept=False)
        coefs = np.asarray(linear_regression.fit(w_for_lr, residuals).coef_)
        return self.from_linear_regression_coefs_to_x0(coefs)
