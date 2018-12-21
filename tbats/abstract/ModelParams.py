import copy
import numpy as np

import tbats.transformation as transformation
from . import ArrayHelper


class ModelParams(object):
    def __init__(self, components, alpha, beta=None, phi=None, boxcox_lambda=None,
                 gamma_params=None,
                 ar_coefs=None, ma_coefs=None, x0=None):
        """
        Represents all parameters needed to calculate model
        :param BATSComponents components: Which parameters the models uses
        :param float alpha:
        :param float|None beta: trend parameter
        :param float|None phi: trend damping parameter
        :param float|None boxcox_lambda: lambda to apply in box-cox transformation
        :param np.array gamma_params: seasonal parameters
        :param np.array ar_coefs: AR parameters for ARMA residuals modelling
        :param np.array ma_coefs: MA parameters for ARMA residuals modelling
        """
        self.components = components
        self.alpha = alpha
        self.boxcox_lambda = None
        if self.components.use_box_cox:
            self.boxcox_lambda = boxcox_lambda
        self.beta = None
        self.phi = None
        if self.components.use_trend:
            self.beta = beta
            self.phi = 1.
            if self.components.use_damped_trend:
                self.phi = phi
        self.gamma_params = self.normalize_gamma_params(gamma_params)

        self.__init_arma(ar_coefs, ma_coefs)

        self.__init_x0(x0)

    @classmethod
    def with_default_starting_params(cls, y, components):
        # abstract method
        raise NotImplementedError()

    def seasonal_components_amount(self):
        # abstract method
        raise NotImplementedError()

    def normalize_gamma_params(self, gamma_params=None):
        # abstract method
        raise NotImplementedError()

    @classmethod
    def find_initial_boxcox_lambda(cls, y, components):
        # Find initial lambda candidate using Guerrero method
        if not components.use_box_cox:
            return None
        return transformation.find_boxcox_lambda(
            y,
            seasonal_periods=components.seasonal_periods,
            bounds=components.box_cox_bounds,
        )

    def with_arma(self, p=0, q=0):
        me = copy.deepcopy(self)
        me.components = self.components.with_arma(p, q)
        me.__init_arma()
        return me

    def with_zero_x0(self):
        me = copy.deepcopy(self)
        me.__init_x0()
        return me

    def with_x0(self, x0):
        me = copy.deepcopy(self)
        me.__init_x0(x0)
        return me

    def with_vector_values(self, vector):
        x0 = self.x0.copy()

        alpha = vector[0]

        offset = 1
        boxcox = None
        if self.components.use_box_cox:
            boxcox = vector[offset]
            # recalculate x0 according to changed box-cox parameter
            x0 = transformation.boxcox(
                transformation.inv_boxcox(x0, self.boxcox_lambda),
                boxcox
            )

            offset += 1

        beta = None
        if self.components.use_trend:
            beta = vector[offset]
            offset += 1

        phi = None
        if self.components.use_damped_trend:  # note that damped trend is used only when trend is used
            phi = vector[offset]
            offset += 1

        gamma_params = vector[offset:(offset + self.components.gamma_params_amount())]  # may be empty
        offset += len(self.components.seasonal_periods)

        ar_coefs = vector[offset:(offset + self.components.p)]  # may be empty
        offset += self.components.p

        ma_coefs = vector[offset:(offset + self.components.q)]  # may be empty
        # offset += self.components.q

        return self.__class__(components=self.components,
                              alpha=alpha, beta=beta, phi=phi, boxcox_lambda=boxcox,
                              gamma_params=gamma_params,
                              ar_coefs=ar_coefs, ma_coefs=ma_coefs, x0=x0)

    def create_x0_of_zeroes(self):
        x0_length = 1
        if self.components.use_trend:
            x0_length += 1

        x0_length += self.seasonal_components_amount()

        x0_length += self.components.arma_length()

        return np.asarray([0] * x0_length)

    def to_vector(self):
        v = [[self.alpha]]
        if self.components.use_box_cox:
            v.append([self.boxcox_lambda])
        if self.components.use_trend:
            v.append([self.beta])
        if self.components.use_damped_trend:
            v.append([self.phi])
        if self.components.gamma_params_amount() > 0:
            v.append(self.gamma_params)
        if self.components.p > 0:
            v.append(self.ar_coefs)
        if self.components.q > 0:
            v.append(self.ma_coefs)
        return np.concatenate(v)

    def amount(self):
        """
        Amount of parameters in the model, including seed state
        :return: int
        """
        amount = 1
        if self.components.use_trend:
            amount += 1
        if self.components.use_damped_trend:
            amount += 1
        if self.components.use_box_cox:
            amount += 1

        return len(self.x0) + amount + self.components.gamma_params_amount() + self.components.arma_length()

    def is_box_cox_in_bounds(self):
        return self.components.is_box_cox_in_bounds(self.boxcox_lambda)

    def summary(self):
        s = self.components.summary() + '\n'
        if not self.boxcox_lambda is None:
            s += 'Box-Cox Lambda %f\n' % self.boxcox_lambda
        s += 'Alpha: %f\n' % self.alpha
        if not self.beta is None:
            s += 'Trend (Beta): %f\n' % self.beta
        if not self.phi is None:
            s += 'Damping Parameter (Phi): %f\n' % self.phi
        s += 'Seasonal Parameters (Gamma): %s\n' % self.gamma_params
        s += 'AR %s\n' % self.ar_coefs
        s += 'MA %s' % self.ma_coefs
        return s

    def __init_arma(self, ar_coefs=None, ma_coefs=None):
        self.ar_coefs = ArrayHelper.to_array(ar_coefs)
        self.ma_coefs = ArrayHelper.to_array(ma_coefs)

        if self.components.p > 0 and len(self.ar_coefs) == 0:
            self.ar_coefs = np.asarray([0] * self.components.p)

        if self.components.q > 0 and len(self.ma_coefs) == 0:
            self.ma_coefs = np.asarray([0] * self.components.q)

    def __init_x0(self, x0=None):
        self.x0 = x0
        if self.x0 is None:
            self.x0 = self.create_x0_of_zeroes()
