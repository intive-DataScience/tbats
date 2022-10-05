import numpy as np

import rpy2.robjects as ro
from rpy2.rinterface_lib.sexp import NULLType
from rpy2.robjects.packages import importr

from tbats.tbats import Model as TBATSModel


class RComparisonBase(object):

    def r_bats(self, y, components):
        components = components.copy()
        if 'seasonal_periods' in components:
            components['seasonal_periods'] = ro.IntVector(components['seasonal_periods'])
        importr('forecast')
        r_bats_func = ro.r['bats']
        r_forecast = ro.r['forecast']
        r_y = ro.FloatVector(list(y))
        r_model = r_bats_func(r_y, **components)
        summary = r_forecast(r_model)
        # predictions = np.array(summary.rx('fitted')).flatten()
        return summary, r_model

    def r_tbats(self, y, components):
        components = components.copy()
        if 'seasonal_periods' in components:
            components['seasonal_periods'] = ro.IntVector(components['seasonal_periods'])
        importr('forecast')
        r_bats_func = ro.r['tbats']
        r_forecast = ro.r['forecast']
        r_y = ro.FloatVector(list(y))
        r_model = r_bats_func(r_y, **components)
        summary = r_forecast(r_model)
        # predictions = np.array(summary.rx('fitted')).flatten()
        return summary, r_model

    def assert_py_model_is_not_worse(self, y, r_summary, r_model, py_model):
        r_predictions = np.asarray(r_summary.rx('fitted')).flatten()
        r_aic = self.get_R_float_or_null(r_model, 'AIC')

        aic = py_model.aic

        if r_aic < aic:
            assert np.isclose(r_aic, aic, atol=0.1)

        r_resid = np.asarray(r_predictions) - np.asarray(y)
        r_mae = np.abs(r_resid).mean()
        mae = np.abs(py_model.resid).mean()

        if r_mae < mae:
            assert np.isclose(r_mae, mae, atol=0.1)

    def assert_forecast_is_not_worse(self, y_to_forecast, r_model, py_model):

        steps = len(y_to_forecast)

        # Forecast from R model
        r_forecast = ro.r['forecast']
        r_predictions = r_forecast(r_model, h=steps)
        r_y_hat = np.array(r_predictions.rx('mean')).flatten()

        # Forecast from python model
        py_y_hat = py_model.forecast(steps)

        r_resid = r_y_hat - y_to_forecast
        py_resid = py_y_hat - y_to_forecast

        r_mae = np.abs(r_resid).mean()
        py_mae = np.abs(py_resid).mean()

        if r_mae < py_mae:
            assert np.isclose(r_mae, py_mae, atol=0.01)

    def compare_model(self, r_summary, r_model, py_model, atol_small=0.001, atol_big=0.01, atol_for_series=0.001):
        r_predictions = np.array(r_summary.rx('fitted')).flatten()

        r_alpha = self.get_R_float_or_null(r_model, 'alpha')
        r_beta = self.get_R_float_or_null(r_model, 'beta')
        r_damping_parameter = self.get_R_float_or_null(r_model, 'damping.parameter')
        r_box_cox_lambda = self.get_R_float_or_null(r_model, 'lambda')
        r_aic = self.get_R_float_or_null(r_model, 'AIC')
        r_gamma = self.get_R_array_or_null(r_model, 'gamma.values')
        r_gamma_1 = self.get_R_array_or_null(r_model, 'gamma.one.values')
        r_gamma_2 = self.get_R_array_or_null(r_model, 'gamma.two.values')
        r_seasonal_periods = self.get_R_array_or_null(r_model, 'seasonal.periods')
        r_ar_coefs = self.get_R_array_or_null(r_model, 'ar.coefficients')
        r_ma_coefs = self.get_R_array_or_null(r_model, 'ma.coefficients')
        r_k_vector = self.get_R_array_or_null(r_model, 'k.vector')

        r_x0 = np.array((r_model.rx('seed.states')[0])).flatten()

        aic = py_model.aic

        # simple parameters comparison
        self.assert_none_or_close(r_alpha, py_model.params.alpha, atol=atol_small)
        self.assert_none_or_close(r_beta, py_model.params.beta, atol=atol_small)
        self.assert_none_or_close(r_damping_parameter, py_model.params.phi, atol=atol_small)
        self.assert_none_or_close(r_box_cox_lambda, py_model.params.box_cox_lambda, atol=atol_small)

        self.assert_array_or_close(r_seasonal_periods, py_model.params.components.seasonal_periods)
        if isinstance(py_model, TBATSModel):
            self.assert_array_or_close(r_k_vector, py_model.params.components.seasonal_harmonics)
            self.assert_array_or_close(r_gamma_1, py_model.params.gamma_1(), atol=atol_big)
            self.assert_array_or_close(r_gamma_2, py_model.params.gamma_2(), atol=atol_big)
        else:
            self.assert_array_or_close(r_gamma, py_model.params.gamma_params)

        self.assert_array_or_close(r_ar_coefs, py_model.params.ar_coefs, atol=atol_big)
        self.assert_array_or_close(r_ma_coefs, py_model.params.ma_coefs, atol=atol_big)

        # seed states comparison
        assert np.allclose(r_x0, py_model.params.x0, atol=atol_small)

        # predictions comparison
        assert np.allclose(r_predictions, py_model.y_hat, atol=atol_for_series)

        # metrics comparison
        assert np.isclose(r_aic, aic, atol=atol_small)

    def compare_forecast(self, r_model, py_model, steps=20, atol=0.01):

        # Forecast from R model
        r_forecast = ro.r['forecast']
        r_predictions = r_forecast(r_model, h=steps)
        r_y_hat = np.array(r_predictions.rx('mean')).flatten()

        # Forecast from python model
        py_y_hat = py_model.forecast(steps)

        assert np.allclose(r_y_hat, py_y_hat, atol=atol)

    def assert_array_or_close(self, r_value, py_value, atol=0.001):
        if r_value is None:
            assert len(py_value) == 0
        else:
            assert np.allclose(r_value, py_value, atol=atol)

    def assert_none_or_close(self, r_value, py_value, atol=0.001):
        if r_value is None:
            assert py_value is None
        else:
            assert np.isclose(r_value, py_value, atol=atol)

    def get_R_float_or_null(self, r_model, paramName):
        val = np.array(r_model.rx(paramName)).flatten()[0]
        if isinstance(val, NULLType):
            return None
        return float(val)

    def get_R_array_or_null(self, r_model, paramName):
        val = np.array(r_model.rx(paramName)).flatten()
        if len(val) == 0 or isinstance(val[0], NULLType):
            return None
        return val
