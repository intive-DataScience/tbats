import numpy as np
import copy

from ..abstract import ArrayHelper, Components as AbstractComponents


class Components(AbstractComponents):
    def __init__(self, use_box_cox=False, use_trend=False, use_damped_trend=False,
                 seasonal_periods=None, seasonal_harmonics=None,
                 use_arma_errors=False, p=0, q=0,
                 box_cox_bounds=(0, 1)):
        """
        Components class contains all the information necessary to determine the amount and the structure
        of parameters used in the model but not parameters' values.

        :param use_box_cox:
        :param use_trend:
        :param use_damped_trend:
        :param list seasonal_periods:
        :param list seasonal_harmonics: If provided its length must match seasonal_periods.
                                        Each entry contains amount of trigonometric harmonics for each season
        :param use_arma_errors:
        :param p:
        :param q:
        """
        super().__init__(
            use_box_cox=use_box_cox, use_trend=use_trend, use_damped_trend=use_damped_trend,
            seasonal_periods=seasonal_periods,
            use_arma_errors=use_arma_errors, p=p, q=q,
            box_cox_bounds=box_cox_bounds)

        self.__init_seasonal_harmonics(seasonal_harmonics)
        # assertion: all provided harmonics are positive, 0 should not be provided

    def gamma_params_amount(self):
        return 2 * len(self.seasonal_periods)

    def with_harmonics_as_ones(self):
        me = copy.deepcopy(self)
        me.__init_seasonal_harmonics()  # sets them to a vector of ones
        return me

    def with_harmonic_for_season(self, season_index, new_harmonic):
        me = copy.deepcopy(self)
        me.seasonal_harmonics[season_index] = new_harmonic
        return me

    def with_seasonal_periods(self, seasonal_periods):
        me = super().with_seasonal_periods(seasonal_periods)
        me.__init_seasonal_harmonics()
        return me

    def without_seasonal_periods(self):
        me = super().without_seasonal_periods()
        me.__init_seasonal_harmonics()
        return me

    def normalize_seasons(self, seasonal_periods):
        return ArrayHelper.to_array(seasonal_periods, float)

    def seasonal_component_lengths(self):
        return self.seasonal_harmonics * 2

    def __init_seasonal_harmonics(self, seasonal_harmonics=None):
        self.seasonal_harmonics = ArrayHelper.to_array(seasonal_harmonics, int)
        if len(self.seasonal_harmonics) != len(self.seasonal_periods):
            self.seasonal_harmonics = np.asarray([1] * len(self.seasonal_periods), int)
