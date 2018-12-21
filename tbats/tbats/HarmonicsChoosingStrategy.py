import numpy as np
import fractions


class HarmonicsChoosingStrategy(object):

    def __init__(self, context):
        self.context = context

    def choose(self, y, components):
        if len(components.seasonal_periods) == 0:
            return np.asarray([])

        components = components.with_harmonics_as_ones()
        best_model = self.context.create_case(components).fit(y)

        max_harmonics = self.calculate_max(components.seasonal_periods)

        for season_index in range(0, len(components.seasonal_periods)):
            best_model = self.choose_for_season(
                season_index=season_index,
                max_harmonic=max_harmonics[season_index],
                best_model_so_far=best_model
            )
        return best_model.params.components.seasonal_harmonics

    def choose_for_season(self, season_index, max_harmonic, best_model_so_far):
        # assertion: best_model_so_far uses harmonics=1 for the seasonality being analysed

        # TODO, this function is sooo ugly

        if max_harmonic == 1:  # nothing to search
            return best_model_so_far

        first_harmonics_to_check = 6
        search_direction = -1  # -1 we will be reducing harmonics, 1 we shall be increasing it

        harmonic_to_check = np.min([max_harmonic, first_harmonics_to_check])

        best_model = best_model_so_far
        best_model_aic = np.inf
        if max_harmonic > first_harmonics_to_check:
            level_model = self.fit_model_like_previous_with_harmonic(
                best_model_so_far, season_index, first_harmonics_to_check
            )
            up_model = self.fit_model_like_previous_with_harmonic(
                best_model_so_far, season_index, first_harmonics_to_check + 1
            )
            down_model = self.fit_model_like_previous_with_harmonic(
                best_model_so_far, season_index, first_harmonics_to_check - 1
            )
            # by default we go into less complex models
            search_direction = -1
            harmonic_to_check = first_harmonics_to_check - 1
            best_model = level_model
            best_model_aic = level_model.aic_

            if down_model.aic_ < best_model.aic_:
                best_model = down_model
                best_model_aic = down_model.aic_
                harmonic_to_check = first_harmonics_to_check - 2

            if up_model.aic_ < best_model.aic_:
                # in such a case we shall go into more complex models
                search_direction = 1
                best_model = up_model
                best_model_aic = up_model.aic_
                harmonic_to_check = first_harmonics_to_check + 2

        while harmonic_to_check > 1 and harmonic_to_check <= max_harmonic:
            candidate_model = self.fit_model_like_previous_with_harmonic(
                best_model_so_far, season_index, harmonic_to_check
            )
            if candidate_model.aic_ > best_model_aic:
                # AIC stopped getting better
                break
            best_model = candidate_model
            best_model_aic = candidate_model.aic_
            harmonic_to_check += search_direction

        if best_model_aic < best_model_so_far.aic_:
            return best_model

        return best_model_so_far

    def fit_model_like_previous_with_harmonic(self, previous_model, season_index, harmonic_to_check):
        components = previous_model.params.components.with_harmonic_for_season(
            season_index=season_index, new_harmonic=harmonic_to_check
        )
        return self.context.create_case(components=components).fit(previous_model.y)

    @classmethod
    def calculate_max(cls, seasonal_periods, dependency_reduction_function=None):
        if dependency_reduction_function is None:
            # should this be used or _better version?
            dependency_reduction_function = cls.max_harmonic_dependency_reduction
        max_harmonics = [1] * len(seasonal_periods)
        for period_index in range(0, len(seasonal_periods)):
            period_length = seasonal_periods[period_index]

            # for 2,3,4 -> 1; for 5,6 -> 2; for 7,8 -> 3; ...
            max_harmonic = np.max([1, int((period_length - 1) / 2)])
            max_harmonic = dependency_reduction_function(
                max_harmonic_proposal=max_harmonic,
                period_length=period_length,
                seasonal_periods=seasonal_periods
            )
            max_harmonics[period_index] = max_harmonic

        return max_harmonics

    @classmethod
    def max_harmonic_dependency_reduction(cls, max_harmonic_proposal, period_length, seasonal_periods):
        if period_length % 1 != 0:  # TODO really?
            # no dependendencies if period length is float
            return max_harmonic_proposal

        shorter_seasonal_periods = seasonal_periods[np.where(seasonal_periods < period_length)]
        max_harmonic = max_harmonic_proposal
        # reduce max harmonic if there is a shorter season
        # that would otherwise repeat the same seasonal trigonometric components
        harmonic = 2
        while harmonic <= max_harmonic:
            if period_length % harmonic == 0:
                amount_of_times_fits_in_period = period_length / harmonic  # this will be integer
                if np.any(shorter_seasonal_periods % amount_of_times_fits_in_period == 0):
                    max_harmonic = harmonic - 1
            harmonic += 1

        return max_harmonic

    @classmethod
    def max_harmonic_dependency_reduction_better(cls, max_harmonic_proposal, period_length, seasonal_periods):
        shorter_seasonal_periods = seasonal_periods[np.where(seasonal_periods < period_length)]
        for shorter_period_length in shorter_seasonal_periods:
            # we need to do type casting here as np.float and np.int64 do not behave well in fractions package
            real_ratio = float(shorter_period_length / period_length)
            # approximate a potentially real number with a rational number
            fraction = fractions.Fraction(real_ratio).limit_denominator(int(period_length) + 1)
            rational_ratio_approximation = fraction.numerator / fraction.denominator

            if not np.isclose(real_ratio, rational_ratio_approximation, atol=1e-5):
                # approximation is bad, no dependency between periods
                continue

            # fraction denominator points to the first harmonic in longer period
            # that shall repeat harmonic in shorter period (being pointed in fraction nominator)
            max_harmonic_proposal = np.min([fraction.denominator - 1, max_harmonic_proposal])

        return max_harmonic_proposal
