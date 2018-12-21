from pmdarima.arima import auto_arima


class Case(object):

    def __init__(self, components, context):
        self.components = components
        self.context = context
        self.model = None  # not fitted yet

    def fit(self, y):
        # TODO what is biasadj?

        best_model = self.fit_initial_model(y)

        if self.components.use_arma_errors:
            # Try adding ARMA to the model
            arma_model = auto_arima(best_model.resid, stationary=True, trend='n',
                                    suppress_warnings=True, error_action='ignore')
            p = arma_model.order[0]
            q = arma_model.order[2]
            if p > 0 or q > 0:  # Found ARMA components
                # Fit model with ARMA errors modelling
                # TODO what if the previous best model is the non-seasonal one?
                # TODO I think we should use non-seasonal model as a base
                model_candidate = self.fit_case(y, self.components.with_arma(p, q))
                if model_candidate.aic_ < best_model.aic_:
                    best_model = model_candidate

        self.model = best_model
        return self.model

    def fit_initial_model(self, y):
        # abstract method
        raise NotImplementedError()

    def fit_case(self, y, case_definition):
        starting_params = self.context.create_default_starting_params(y, components=case_definition)
        return self.fit_with_starting_params(y, starting_params)

    def fit_with_starting_params(self, y, model_params):
        optimization = self.context.create_params_optimizer()
        optimization = optimization.optimize(y, model_params)
        model = optimization.optimal_model()
        if not optimization.converged():
            model.add_warning("Optimization did not converge")
        return model
