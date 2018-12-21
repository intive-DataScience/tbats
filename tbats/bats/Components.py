from ..abstract import ArrayHelper, Components as AbstractComponents


class Components(AbstractComponents):
    def __init__(self, use_box_cox=False, use_trend=False, use_damped_trend=False,
                 seasonal_periods=None, use_arma_errors=False, p=0, q=0,
                 box_cox_bounds=(0, 1)):
        super().__init__(
            use_box_cox=use_box_cox, use_trend=use_trend, use_damped_trend=use_damped_trend,
            seasonal_periods=seasonal_periods,
            use_arma_errors=use_arma_errors, p=p, q=q,
            box_cox_bounds=box_cox_bounds)

    @classmethod
    def create_constant_components(cls):
        return cls(use_box_cox=False, use_trend=False, seasonal_periods=None, use_arma_errors=False)

    def seasonal_component_lengths(self):
        return self.seasonal_periods

    def normalize_seasons(self, seasonal_periods):
        # Ensure seasons are integers
        return ArrayHelper.to_array(seasonal_periods, int)

    def gamma_params_amount(self):
        return len(self.seasonal_periods)
