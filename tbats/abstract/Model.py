import numpy as np
from sklearn.utils.validation import check_array, column_or_1d as c1d
from numpy.linalg import LinAlgError
import warnings

import tbats.error as error
import tbats.transformation as transformation


class Model(object):
    def __init__(self, model_params, context, validate_input=True):
        self.context = context
        self.warnings = []
        self.params = model_params
        self.validate_input = validate_input
        self.matrix = self.context.create_matrix_builder(self.params)

        self.is_fitted = False
        self.y = None
        self.y_hat = None
        self.resid_boxcox = None
        self.resid = None
        self.x_last = None
        self.aic_ = np.inf

    def likelihood(self):
        if not self.is_fitted:
            return np.inf

        residuals = self.resid_boxcox

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                likelihood_part = len(residuals) * np.log(np.sum(residuals * residuals))
            except RuntimeWarning:
                # calculation issues, values close to max float value
                return np.inf

        boxcox_part = 0
        if self.params.components.use_box_cox:
            boxcox_part = 2 * (self.params.boxcox_lambda - 1) * np.sum(np.log(self.y))

        return likelihood_part - boxcox_part

    def aic(self):
        likelihood = self.likelihood()
        if likelihood == np.inf:
            return np.inf
        return likelihood + 2 * self.params.amount()

    def forecast(self, steps=1):
        if not self.is_fitted:
            self.context.get_exception_handler().exception(
                'Model must be fitted to be able to forecast. Use fit method first.',
                error.BatsException
            )
        steps = int(steps)
        if steps < 1:
            self.context.get_exception_handler().exception(
                'Parameter \'steps\' must be a positive integer',
                error.InputArgsException
            )

        F = self.matrix.make_F_matrix()
        w = self.matrix.make_w_vector()

        # TODO add confidence intervals

        # initialize matrices
        # next_x = x[:, len(self.y)]
        yw_hat = np.asarray([0.0] * steps)
        x = self.x_last

        for t in range(0, steps):
            yw_hat[t] = w @ x
            x = F @ x

        y_hat = self.inv_boxcox(yw_hat)
        return y_hat

    def fit_ORIGINAL(self, y):
        self.warnings = []
        self.is_fitted = False
        params = self.params

        if self.validate_input:
            try:
                y = c1d(check_array(y, ensure_2d=False, force_all_finite=True, ensure_min_samples=1,
                                    copy=True, dtype=np.float64))  # type: np.ndarray
            except Exception as validation_exception:
                self.context.get_exception_handler().exception("y series is invalid",
                                                               error.InputArgsException,
                                                               previous_exception=validation_exception)

        yw = self.boxcox(y)

        matrix_builder = self.matrix
        w = matrix_builder.make_w_vector()
        g = matrix_builder.make_g_vector().T
        F = matrix_builder.make_F_matrix()

        # initialize matrices
        yw_hat = np.zeros((1, len(y)))
        x = np.matrix(np.zeros((len(params.x0), len(yw) + 1)))
        x[:, 0] = np.matrix(params.x0).T
        e = np.zeros((1, len(y)))

        # calculation, please note x[0] holds x0,
        # therefore index is shifted by one from the original mathematical equation
        for t in range(0, len(y)):
            yw_hat[:, t] = w * x[:, t]
            e[:, t] = yw[t] - yw_hat[:, t]
            x[:, t + 1] = F * x[:, t] + g * e[:, t]

        # store fit results
        self.y = self.inv_boxcox(yw)
        self.y_hat = self.inv_boxcox(yw_hat[0, :])
        self.resid_boxcox = e[0, :]  # box-cox residuals, when no box-cox equal to self.resid
        self.resid = self.y - self.y_hat
        self.x = x

        self.is_fitted = True
        self.aic_ = self.aic()
        return self

    def fit(self, y):
        self.warnings = []
        self.is_fitted = False
        params = self.params

        if self.validate_input:
            try:
                y = c1d(check_array(y, ensure_2d=False, force_all_finite=True, ensure_min_samples=1,
                                    copy=True, dtype=np.float64))  # type: np.ndarray
            except Exception as validation_exception:
                self.context.get_exception_handler().exception("y series is invalid",
                                                               error.InputArgsException,
                                                               previous_exception=validation_exception)

        yw = self.boxcox(y)

        matrix_builder = self.matrix
        w = matrix_builder.make_w_vector()
        g = matrix_builder.make_g_vector()
        F = matrix_builder.make_F_matrix()

        # initialize matrices
        yw_hat = np.asarray([0.0] * len(y))
        # x = np.matrix(np.zeros((len(params.x0), len(yw) + 1)))
        x = params.x0

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                for t in range(0, len(y)):
                    yw_hat[t] = w @ x
                    e = yw[t] - yw_hat[t]
                    x = F @ x + g * e
            except RuntimeWarning:
                # calculation issues, values close to max float value
                self.is_fitted = False
                return self

        # store fit results
        self.x_last = x
        self.resid_boxcox = yw - yw_hat
        self.y = self.inv_boxcox(yw)
        self.y_hat = self.inv_boxcox(yw_hat)
        self.resid = self.y - self.y_hat

        self.is_fitted = True
        self.aic_ = self.aic()
        return self

    def boxcox(self, y):
        yw = y
        if self.params.components.use_box_cox:
            yw = transformation.boxcox(y, lam=self.params.boxcox_lambda)
        return yw

    def inv_boxcox(self, yw):
        y = yw
        if self.params.components.use_box_cox:
            y = transformation.inv_boxcox(yw, lam=self.params.boxcox_lambda)
        return y

    def can_be_admissible(self):
        if not self.params.is_box_cox_in_bounds():
            return False

        params = self.params
        if params.components.use_damped_trend and (params.phi < 0.8 or params.phi > 1):
            return False

        if not self.__AR_is_stationary(params.ar_coefs):
            return False

        if not self.__MA_is_invertible(params.ma_coefs):
            return False

        D = self.matrix.calculate_D_matrix()
        return self.__D_matrix_eigen_values_check(D)

    def is_admissible(self):
        if not self.is_fitted:
            return False
        return self.can_be_admissible()

    @staticmethod
    def __AR_is_stationary(ar_coefs):
        # cut out trailing and non-significant AR components
        significant_indices = np.where(np.abs(ar_coefs) > 1e-08)[0]
        if len(significant_indices) == 0:
            return True
        p = np.max(significant_indices) + 1
        significant_ar_coefs = ar_coefs[0:p]

        roots = np.polynomial.polynomial.polyroots(np.concatenate([[1], -significant_ar_coefs]))
        # Note that np.abs also works with complex numbers. It provides length of complex number
        # There should be no roots in the unit circle
        return np.all(np.abs(roots) > 1.0)

    @staticmethod
    def __MA_is_invertible(ma_coefs):
        # cut out trailing and non-significant AR components
        significant_indices = np.where(np.abs(ma_coefs) > 1e-08)[0]
        if len(significant_indices) == 0:
            return True
        q = np.max(significant_indices) + 1
        significant_ma_coefs = ma_coefs[0:q]

        roots = np.polynomial.polynomial.polyroots(np.concatenate([[1], significant_ma_coefs]))
        # Note that np.abs also works with complex numbers. It provides length of complex number
        # There should be no roots in the unit circle
        return np.all(np.abs(roots) > 1.0)

    @staticmethod
    def __D_matrix_eigen_values_check(D):
        try:
            eigen_values = np.linalg.eigvals(D)
        except LinAlgError:
            return False
        return np.all(np.abs(eigen_values) < 1.01)

    def add_warning(self, message):
        self.warnings.append(message)

    def summary(self):
        str = ''
        str += self.params.summary() + '\n'
        str += 'AIC %f' % self.aic_
        return str
