import numpy as np

from ..abstract import Estimator
from . import Context


class TBATS(Estimator):
    def __init__(self, use_box_cox=None, use_trend=None, use_damped_trend=None,
                 seasonal_periods=None, use_arma_errors=True,
                 show_warnings=True, context=None):
        if context is None:
            context = Context(show_warnings)  # the default BATS context
        super().__init__(context, use_box_cox=use_box_cox, use_trend=use_trend, use_damped_trend=use_damped_trend,
                         seasonal_periods=seasonal_periods, use_arma_errors=use_arma_errors)

    def normalize_seasonal_periods(self, seasonal_periods):
        return self.normalize_seasonal_periods_to_type(seasonal_periods, dtype=float)

    def do_fit(self, y):
        components_grid = self.prepare_non_seasonal_components_grid()
        non_seasonal_model = self.choose_model_from_possible_component_settings(y, components_grid=components_grid)

        harmonics_choosing_strategy = self.context.create_harmonics_choosing_strategy()
        chosen_harmonics = harmonics_choosing_strategy.choose(y, self.create_most_complex_components())
        components_grid = self.prepare_components_grid(seasonal_harmonics=chosen_harmonics)
        seasonal_model = self.choose_model_from_possible_component_settings(y, components_grid=components_grid)

        if non_seasonal_model.aic_ < seasonal_model.aic_:
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

    def choose_model_from_possible_component_settings(self, y, components_grid):
        best_model = None
        best_model_aic = np.inf
        for components_combination in components_grid:
            case = self.context.create_case_from_dictionary(**components_combination)
            model = case.fit(y)
            if model.aic_ < best_model_aic:
                best_model_aic = model.aic_
                best_model = model
        return best_model
