import numpy as np

from ..abstract import Estimator
from . import Context


class TBATS(Estimator):
    def __init__(self, use_box_cox=None, use_trend=None, use_damped_trend=None,
                 seasonal_periods=None, use_arma_errors=True,
                 show_warnings=True, context=None,
                 n_jobs=None):
        if context is None:
            context = Context(show_warnings)  # the default TBATS context
        super().__init__(context, use_box_cox=use_box_cox, use_trend=use_trend, use_damped_trend=use_damped_trend,
                         seasonal_periods=seasonal_periods, use_arma_errors=use_arma_errors,
                         n_jobs=n_jobs)

    def normalize_seasonal_periods(self, seasonal_periods):
        return self.normalize_seasonal_periods_to_type(seasonal_periods, dtype=float)

    def do_fit(self, y):
        components_grid = self.prepare_non_seasonal_components_grid()
        non_seasonal_model = self.choose_model_from_possible_component_settings(y, components_grid=components_grid)

        harmonics_choosing_strategy = self.context.create_harmonics_choosing_strategy(n_jobs=self.n_jobs)
        chosen_harmonics = harmonics_choosing_strategy.choose(y, self.create_most_complex_components())
        components_grid = self.prepare_components_grid(seasonal_harmonics=chosen_harmonics)
        seasonal_model = self.choose_model_from_possible_component_settings(y, components_grid=components_grid)

        if non_seasonal_model.aic < seasonal_model.aic:
            return non_seasonal_model

        return seasonal_model

    def create_most_complex_components(self):
        components = dict(
            use_box_cox=self.use_box_cox is not False,
            use_trend=self.use_trend is not False,
            use_damped_trend=self.use_damped_trend is not False,
            seasonal_periods=self.seasonal_periods,
            use_arma_errors=False,
        )
        return self.context.create_components(**components)
