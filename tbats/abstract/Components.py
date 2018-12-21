import copy
import numpy as np


class Components(object):
    def __init__(self, use_box_cox=False, use_trend=False, use_damped_trend=False,
                 seasonal_periods=None, use_arma_errors=False, p=0, q=0,
                 box_cox_bounds=(0, 1)):
        """
        Components class contains all the information necessary to determine the amount and the structure
        of parameters used in the model but not parameters' values.

        :param use_box_cox:
        :param use_trend:
        :param use_damped_trend:
        :param seasonal_periods: Array of seasons' lengths. Each entry should contain amount of observations per season.
        :param use_arma_errors:
        :param p:
        :param q:
        """
        self.use_box_cox = use_box_cox
        self.box_cox_bounds = box_cox_bounds
        self.use_trend = use_trend
        self.use_damped_trend = False
        if self.use_trend:  # only when trend is used
            self.use_damped_trend = use_damped_trend
        self.seasonal_periods = self.normalize_seasons(seasonal_periods)
        self.__use_arma(do_use=use_arma_errors, p=p, q=q)

    @classmethod
    def create_constant_components(cls):
        # abstract method
        raise NotImplementedError()

    def normalize_seasons(self, seasonal_periods):
        # abstract method
        raise NotImplementedError()

    def seasonal_component_lengths(self):
        # abstract method
        raise NotImplementedError()

    def gamma_params_amount(self):
        # abstract method
        raise NotImplementedError()

    def arma_length(self):
        return self.p + self.q

    def with_seasonal_periods(self, seasonal_periods):
        components = copy.deepcopy(self)
        components.seasonal_periods = self.normalize_seasons(seasonal_periods)
        return components

    def without_seasonal_periods(self):
        components = copy.deepcopy(self)
        components.seasonal_periods = np.asarray([])
        return components

    def without_arma(self):
        components = copy.deepcopy(self)
        return components.__use_arma(False)

    def with_arma(self, p=0, q=0):
        components = copy.deepcopy(self)
        return components.__use_arma(do_use=True, p=p, q=q)

    def is_box_cox_in_bounds(self, box_cox_lambda):
        if self.use_box_cox:
            return self.box_cox_bounds[0] <= box_cox_lambda and self.box_cox_bounds[1] >= box_cox_lambda
        return True

    def __use_arma(self, do_use=True, p=0, q=0):
        self.use_arma_errors = do_use
        self.p = 0
        self.q = 0
        if do_use:  # only when ARMA errors are used
            self.p = p
            self.q = q
        return self

    def summary(self):
        s = "Use Box-Cox: %r\n" % self.use_box_cox
        s += "Use trend: %r\n" % self.use_trend
        s += "Use damped trend: %r\n" % self.use_damped_trend
        s += "Seasonal periods: %s\n" % self.seasonal_periods
        s += "ARMA errors (p, q): (%d, %d)" % (self.p, self.q)
        return s
