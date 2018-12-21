import numpy as np

from ..abstract import Estimator
from . import Context


class BATS(Estimator):
    def __init__(self, use_box_cox=None, use_trend=None, use_damped_trend=None,
                 seasonal_periods=None, use_arma_errors=True,
                 show_warnings=True, context=None):
        """

        :param use_box_cox: True/False/None
        :param use_trend:
        :param use_damped_trend:
        :param seasonal_periods:
        :param use_arma_errors:
        :param show_warnings:
        :param context: Provide this to override default behaviors, see abstract.ContextInterface
        """
        if context is None:
            context = Context(show_warnings)  # the default BATS context
        super().__init__(context, use_box_cox=use_box_cox, use_trend=use_trend, use_damped_trend=use_damped_trend,
                         seasonal_periods=seasonal_periods, use_arma_errors=use_arma_errors)

    def normalize_seasonal_periods(self, seasonal_periods):
        return self.normalize_seasonal_periods_to_type(seasonal_periods, dtype=int)

    def do_fit(self, y):
        params_grid = self.prepare_components_grid()
        best_model = None
        best_model_aic = np.inf
        for params_combination in params_grid:
            case = self.context.create_case_from_dictionary(**params_combination)
            model = case.fit(y)
            if model.aic_ < best_model_aic:
                best_model_aic = model.aic_
                best_model = model
        return best_model
