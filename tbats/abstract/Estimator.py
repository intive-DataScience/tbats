import numpy as np
import multiprocessing
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, column_or_1d as c1d
from sklearn.model_selection import ParameterGrid

import tbats.error as error


class Estimator(BaseEstimator):
    def __init__(self, context, use_box_cox=None, use_trend=None, use_damped_trend=None,
                 seasonal_periods=None, use_arma_errors=True,
                 n_jobs=None):
        self.context = context
        self.n_jobs = n_jobs

        self.seasonal_periods = self.normalize_seasonal_periods(seasonal_periods)
        self.use_box_cox = use_box_cox
        self.use_arma_errors = use_arma_errors
        self.use_trend = use_trend
        if use_trend is False:
            if use_damped_trend is True:
                self.context.get_exception_handler().warn(
                    "When use_damped_trend can be used only with use_trend. Setting damped trend to False.",
                    error.InputArgsWarning
                )
            use_damped_trend = False
        self.use_damped_trend = use_damped_trend

    def normalize_seasonal_periods(self, seasonal_periods):
        # abstract method
        raise NotImplementedError()

    def do_fit(self, y):
        # abstract method
        raise NotImplementedError()

    def fit(self, y):
        y = self.validate(y)
        if y is False:
            # Input data is not valid and no exception was raised yet.
            # This can happen only when one overrides default exception handler (see tbats.error.ExceptionHandler)
            return None

        if np.allclose(y, y[0]):
            return self.context.create_constant_model(y[0]).fit(y)

        best_model = self.do_fit(y)

        for warning in best_model.warnings:
            self.context.get_exception_handler().warn(warning, error.ModelWarning)

        return best_model

    def validate(self, y):
        try:
            y = c1d(check_array(y, ensure_2d=False, force_all_finite=True, ensure_min_samples=1,
                                copy=True, dtype=np.float64))  # type: np.ndarray
        except Exception as validation_exception:
            self.context.get_exception_handler().exception(
                "y series is invalid", error.InputArgsException, previous_exception=validation_exception
            )
            return False

        if np.any(y <= 0):
            if self.use_box_cox is True:
                self.context.get_exception_handler().warn(
                    "Box-Cox transformation (use_box_cox) was forced to True "
                    "but there are negative values in input series. "
                    "Setting use_box_cox to False.",
                    error.InputArgsWarning
                )
            self.use_box_cox = False

        return y

    # TODO remove when parallel code is ready
    def choose_model_from_possible_component_settings_serial(self, y, components_grid):
        best_model = None
        best_model_aic = np.inf
        for components_combination in components_grid:
            case = self.context.create_case_from_dictionary(**components_combination)
            model = case.fit(y)
            if model.aic < best_model_aic:
                best_model_aic = model.aic
                best_model = model
        return best_model

    def _case_fit(self, components_combination):
        case = self.context.create_case_from_dictionary(**components_combination)
        return case.fit(self._y)

    def choose_model_from_possible_component_settings(self, y, components_grid):
        self._y = y
        # note n_jobs = None means to use cpu_count()
        pool = multiprocessing.pool.Pool(processes=self.n_jobs)
        models = pool.map(self._case_fit, components_grid)
        self._y = None

        best_model = models[0]
        for model in models:
            if model.aic < best_model.aic:
                best_model = model
        return best_model

    def prepare_components_grid(self, seasonal_harmonics=None):
        allowed_combinations = []

        use_box_cox = self.use_box_cox

        base_combination = {
            'use_box_cox': self.__prepare_component_boolean_combinations(use_box_cox),
            'use_arma_errors': [self.use_arma_errors],
            'seasonal_periods': [self.seasonal_periods],
        }
        if seasonal_harmonics is not None:
            base_combination['seasonal_harmonics'] = [seasonal_harmonics]

        if self.use_trend is not True:  # False or None
            allowed_combinations.append({
                **base_combination,
                **{
                    'use_trend': [False],
                    'use_damped_trend': [False],  # Damped trend must be False when trend is False
                }
            })

        if self.use_trend is not False:  # True or None
            allowed_combinations.append({
                **base_combination,
                **{
                    'use_trend': [True],
                    'use_damped_trend': self.__prepare_component_boolean_combinations(self.use_damped_trend),
                }
            })
        return ParameterGrid(allowed_combinations)

    def prepare_non_seasonal_components_grid(self):
        allowed_combinations = []

        use_box_cox = self.use_box_cox

        base_combination = {
            'use_box_cox': self.__prepare_component_boolean_combinations(use_box_cox),
            'use_arma_errors': [self.use_arma_errors],
        }

        if self.use_trend is not True:  # False or None
            allowed_combinations.append({
                **base_combination,
                **{
                    'use_trend': [False],
                    'use_damped_trend': [False],  # Damped trend must be False when trend is False
                }
            })

        if self.use_trend is not False:  # True or None
            allowed_combinations.append({
                **base_combination,
                **{
                    'use_trend': [True],
                    'use_damped_trend': self.__prepare_component_boolean_combinations(self.use_damped_trend),
                }
            })
        return ParameterGrid(allowed_combinations)

    @staticmethod
    def __prepare_component_boolean_combinations(param):
        combinations = [param]
        if param is None:
            combinations = [False, True]
        return combinations

    def normalize_seasonal_periods_to_type(self, seasonal_periods, dtype):
        if seasonal_periods is not None:
            try:
                seasonal_periods = c1d(check_array(seasonal_periods, ensure_2d=False, force_all_finite=True,
                                                   ensure_min_samples=0,
                                                   copy=True, dtype=dtype))
            except Exception as validation_exception:
                self.context.get_exception_handler().exception("seasonal_periods definition is invalid",
                                                               error.InputArgsException,
                                                               previous_exception=validation_exception)

            seasonal_periods = np.unique(seasonal_periods)
            if len(seasonal_periods[np.where(seasonal_periods <= 1)]) > 0:
                self.context.get_exception_handler().warn(
                    "All seasonal periods should be integer values greater than 1. "
                    "Ignoring all seasonal period values that do not meet this condition.",
                    error.InputArgsWarning
                )
            seasonal_periods = seasonal_periods[np.where(seasonal_periods > 1)]
            seasonal_periods.sort()
            if len(seasonal_periods) == 0:
                seasonal_periods = None
        return seasonal_periods
