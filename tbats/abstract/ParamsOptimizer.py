import numpy as np
from scipy.optimize import minimize


class ParamsOptimizer(object):

    def __init__(self, context):
        self.context = context

        self.starting_params = None

        # Initialize optimization params
        self.success = False  # did not converge yet
        self.likelihood = np.inf
        self.optimal_params = None
        self.y = None

    def converged(self):
        return self.success

    def optimal_model(self):
        # will return a model even if optimization did not converge
        model = self.context.create_model(self.optimal_params).fit(self.y)
        if not model.is_fitted:
            model.add_warning("Model did not calculate properly and seems unusable.")
        if not model.can_be_admissible():
            model.add_warning("Model is not admissible! Forecasts may be unstable. Check long term forecasts.")
        return model

    def optimize(self, y, initial_params, calculate_seed=True):
        self.y = y

        # Initialize seed using linear regression
        self.starting_params = initial_params
        if calculate_seed:
            x0 = self.calculate_seed_x0(y, initial_params)
            self.starting_params = initial_params.with_x0(x0)

        starting_vector = self.starting_params.to_vector()
        result = minimize(
            self.scale_and_calculate_likelihood,  # function being optimized
            x0=self.inv_scale_vector(starting_vector),
            # self.calculate_likelihood,
            # x0=starting_vector,
            method='Nelder-Mead',
            options={
                'maxiter': 100 * (len(starting_vector) ** 2),
                'fatol': 1e-8,
                # 'disp': True,
            }
        )
        self.success = result.success

        self.likelihood = result.fun
        self.optimal_params = self.starting_params.with_vector_values(
            self.scale_vector(result.x)
        )

        return self

    def scale_and_calculate_likelihood(self, optimization_vector):
        optimization_vector = self.scale_vector(optimization_vector)
        return self.calculate_likelihood(optimization_vector)

    def calculate_likelihood(self, optimization_vector):
        infinity = 10 ** 10  # we can not return np.inf as optimization will not work
        # print(optimization_vector)
        params = self.starting_params.with_vector_values(optimization_vector)
        model = self.context.create_model(params, validate_input=False)
        if not model.can_be_admissible():  # don't even calculate the model for such params
            return infinity
        model = model.fit(self.y)
        likelihood = model.likelihood()
        if likelihood == np.inf:
            return infinity
        return likelihood

    def calculate_seed_x0(self, y, params):
        model = self.context.create_model(params.with_zero_x0(), validate_input=False)
        y_tilda = model.fit(y).resid_boxcox

        w = model.matrix.make_w_vector()

        D = model.matrix.calculate_D_matrix()

        w_tilda = np.zeros((len(y), len(w)))
        w_tilda[0, :] = w

        D_transposed = D.T
        for t in range(1, len(y)):
            w_tilda[t, :] = D_transposed @ w_tilda[t - 1, :]

        # TODO I am wondering, since we are removing this ARMA part, do we need to calculate it

        # linear regression to find x0
        seed_finder = self.context.create_seed_finder(params.components)
        return seed_finder.find(w_tilda, y_tilda)

    def scale_vector(self, vector):
        scale = self.get_scale()
        return vector * scale

    def inv_scale_vector(self, vector):
        scale = self.get_scale()
        return vector / scale

    def get_scale(self):
        components = self.starting_params.components
        scale = [[0.01]]
        if components.use_box_cox:
            scale.append([0.001])
        if components.use_trend:
            scale.append([0.01])
            if components.use_damped_trend:
                scale.append([0.01])
        if components.gamma_params_amount() > 0:
            scale.append([1e-5] * components.gamma_params_amount())
        if components.arma_length():
            scale.append([0.1] * components.arma_length())
        return np.concatenate(scale)
