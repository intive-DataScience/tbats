from ..abstract import Case as AbstractCase


class Case(AbstractCase):

    def fit_initial_model(self, y):
        model = self.fit_case(y, self.components.without_arma())
        if len(self.components.seasonal_periods) > 0:
            # Try non-seasonal model without ARMA
            model_candidate = self.fit_case(y, self.components.without_seasonal_periods().without_arma())
            if model_candidate.aic_ < model.aic_:
                model = model_candidate
        return model
