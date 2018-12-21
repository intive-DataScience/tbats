from ..abstract import Case as AbstractCase


class Case(AbstractCase):

    def fit_initial_model(self, y):
        # TBATS, unlike BATS, does not check for non-seasonal model,
        # non-seasonal models for TBATS are checked when seasonal harmonics are being determined
        return self.fit_case(y, self.components.without_arma())
