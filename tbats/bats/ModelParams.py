import numpy as np

import tbats.transformation as transformation
from ..abstract import ArrayHelper, ModelParams as AbstractModelParams


class ModelParams(AbstractModelParams):

    def __init__(self, components, alpha, beta=None, phi=None, boxcox_lambda=None,
                 gamma_params=None,
                 ar_coefs=None, ma_coefs=None, x0=None):
        super().__init__(components=components, alpha=alpha, beta=beta, phi=phi, boxcox_lambda=boxcox_lambda,
                         gamma_params=gamma_params, ar_coefs=ar_coefs, ma_coefs=ma_coefs, x0=x0)

    @classmethod
    def with_default_starting_params(cls, y, components):
        """
        Factory method for starting model parameters
        :param BATSComponents components: Which parameters the models uses
        :return: BATSModelParams
        """
        alpha = 0.09
        if components.seasonal_periods.sum() > 16:
            alpha = 1e-6

        beta = None
        phi = None
        if components.use_trend:
            beta = 0.05
            if components.seasonal_periods.sum() > 16:
                beta = 5e-7
            phi = 1
            if components.use_damped_trend:
                phi = 0.999

        gamma_params = None
        if len(components.seasonal_periods) > 0:
            gamma_params = [0.001] * components.gamma_params_amount()

        boxcox_lambda = cls.find_initial_boxcox_lambda(y, components)

        # note that ARMA and x0 will be initialized to 0's in constructor
        return cls(components=components,
                   alpha=alpha, beta=beta, phi=phi,
                   gamma_params=gamma_params,
                   boxcox_lambda=boxcox_lambda)

    def normalize_gamma_params(self, gamma_params=None):
        gamma_params = ArrayHelper.to_array(gamma_params)
        if len(gamma_params) != len(self.components.seasonal_periods):
            gamma_params = np.asarray([0.0] * len(self.components.seasonal_periods), float)
        return gamma_params

    def seasonal_components_amount(self):
        return int(self.components.seasonal_periods.sum())
