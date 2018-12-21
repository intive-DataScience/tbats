import pytest
import numpy as np

from tbats.tbats import HarmonicsChoosingStrategy, Context, Components, ModelParams


class TestTBATSHarmonicsChoosingStrategy(object):
    class ModelMock:
        def __init__(self, y, params, aic_score):
            self.params = params
            self.aic_score = aic_score
            self.y = y
            self.aic_ = aic_score
            self.is_fitted = True

        def aic(self):
            return self.aic_score

    class CaseMock:
        def __init__(self, components, aic_score):
            self.components = components
            self.aic_score = aic_score

        def fit(self, y):
            params = ModelParams(self.components, alpha=0)
            return TestTBATSHarmonicsChoosingStrategy.ModelMock(y, params, self.aic_score)

    class ContextMock:
        def __init__(self, aic_score_map):
            self.aic_score_map = aic_score_map
            pass

        def create_case(self, components):
            for (harmonics, aic_score) in self.aic_score_map:
                if np.array_equal(components.seasonal_harmonics, harmonics):
                    return TestTBATSHarmonicsChoosingStrategy.CaseMock(components, aic_score)
            raise Exception('Unknown score for harmonics ' + str(components.seasonal_harmonics))

    @pytest.mark.parametrize(
        "components, aic_score_map, expected_harmonics",
        [
            [  # no periods means no harmonics
                dict(),
                [
                    # strategy shouldn't even attempt to build a model
                ],
                [],
            ],
            [  # when score for 5 is better than for 6, strategy will be checking less complex models
                dict(seasonal_periods=[21]),
                [  # AIC score map for chosen harmonics
                    ([1], 10.),
                    ([6], 8.),
                    ([7], 9.),
                    ([5], 7.),
                    ([4], 6.),  # best model
                    ([3], 7.),
                ],
                [4],
            ],
            [  # when score for 7 is better than one for 6, strategy will be checking more complex models
                dict(seasonal_periods=[21]),
                [  # AIC score map for chosen harmonics
                    ([1], 20.),
                    ([5], 9.),
                    ([6], 8.),
                    ([7], 7.),
                    ([8], 6.),
                    ([9], 5.),  # best model
                    ([10], 6.),
                ],
                [9],
            ],
            [  # if initial model is the best one, it should be returned
                dict(seasonal_periods=[30]),
                [  # AIC score map for chosen harmonics
                    ([1], 1.),  # best model
                    ([5], 8.),
                    ([6], 6.),
                    ([7], 9.),
                ],
                [1],
            ],
            [  # a model with small amount of harmonics
                dict(seasonal_periods=[4]),
                [  # AIC score map for chosen harmonics
                    ([1], 1.),  # this is the only possible model
                ],
                [1],
            ],
            [  # two periods
                dict(seasonal_periods=[7, 365]),
                [  # AIC score map for chosen harmonics
                    ([1, 1], 20.),
                    ([2, 1], 18.),  # best model for 1st season
                    ([3, 1], 19.),
                    ([2, 5], 16.),
                    ([2, 6], 15.),
                    ([2, 7], 14.),
                    ([2, 8], 13.),  # best model
                    ([2, 9], 16.),
                ],
                [2, 8],
            ],
            [  # three periods
                dict(seasonal_periods=[7, 30, 365]),
                [  # AIC score map for chosen harmonics
                    ([1, 1, 1], 20.),
                    ([2, 1, 1], 18.),
                    ([3, 1, 1], 17.),  # best model for 1st season
                    ([4, 1, 1], 19.),
                    ([3, 5, 1], 16.),
                    ([3, 6, 1], 15.),
                    ([3, 7, 1], 14.),
                    ([3, 8, 1], 13.),  # best model for 2nd season
                    ([3, 9, 1], 16.),
                    ([3, 8, 7], 14.),
                    ([3, 8, 6], 12.),
                    ([3, 8, 5], 11.),
                    ([3, 8, 4], 10.),  # best model
                    ([3, 8, 3], 11.),
                ],
                [3, 8, 4],
            ],
        ]
    )
    def test_choose(self, components, aic_score_map, expected_harmonics):
        strategy = HarmonicsChoosingStrategy(self.ContextMock(aic_score_map))
        harmonics = strategy.choose([1, 2, 3], Components(**components))
        assert np.array_equal(expected_harmonics, harmonics)

    @pytest.mark.parametrize(
        "seasonal_periods, expected_max_harmonics",
        [
            [  # no periods means no harmonics
                [], [],
            ],
            [  # period of length 2 is limited to 1
                [2], [1],
            ],
            [  # floor((4 - 1) / 2) = 1
                [4], [1],
            ],
            [
                [5], [2],
            ],
            [
                [9], [4],
            ],
            [  # floor((28 - 1) / 2) = 13
                [28], [13],
            ],
            [  # 2nd seasonal harmonic for 16 is equal to 1st seasonal harmonic for 8
                [8, 16], [3, 1]
            ],
            [  # 10th seasonal harmonic for 100 is equal to 1st seasonal harmonic for 10
                [10, 100], [4, 9]
            ],
            [  # 2nd seasonal harmonic for 100 is equal to 1st seasonal harmonic for 50
                # 5th seasonal harmonic for 50 is equal to 1st seasonal harmonic for 10
                [10, 50, 100], [4, 4, 1]
            ],
            [  # 5th seasonal harmonic for 50 is equal to 2nd seasonal harmonic for 20
                [20, 50], [9, 4]
            ],
            [  # This method does not work with floats, see _better implementation
                [25.5, 51], [12, 25]
            ],
        ]
    )
    def test_calculate_max(self, seasonal_periods, expected_max_harmonics):
        strategy = HarmonicsChoosingStrategy(Context())
        harmonics = strategy.calculate_max(np.array(seasonal_periods))
        assert np.array_equal(expected_max_harmonics, harmonics)

    @pytest.mark.parametrize(
        "seasonal_periods, expected_max_harmonics",
        [
            [  # no periods means no harmonics
                [], [],
            ],
            [  # period of length 2 is limited to 1
                [2], [1],
            ],
            [  # floor((4 - 1) / 2) = 1
                [4], [1],
            ],
            [
                [5], [2],
            ],
            [
                [9], [4],
            ],
            [  # floor((28 - 1) / 2) = 13
                [28], [13],
            ],
            [  # 2nd seasonal harmonic for 16 is equal to 1st seasonal harmonic for 8
                [8, 16], [3, 1]
            ],
            [  # 10th seasonal harmonic for 100 is equal to 1st seasonal harmonic for 10
                [10, 100], [4, 9]
            ],
            [  # 2nd seasonal harmonic for 100 is equal to 1st seasonal harmonic for 50
                # 5th seasonal harmonic for 50 is equal to 1st seasonal harmonic for 10
                [10, 50, 100], [4, 4, 1]
            ],
            [  # 5th seasonal harmonic for 50 is equal to 2nd seasonal harmonic for 20
                [20, 50], [9, 4]
            ],
            [  # The better method also works with floats
                [25.5, 51], [12, 1]
            ],
        ]
    )
    def test_calculate_max_better(self, seasonal_periods, expected_max_harmonics):
        strategy = HarmonicsChoosingStrategy(Context())
        harmonics = strategy.calculate_max(
            np.asarray(seasonal_periods),
            HarmonicsChoosingStrategy.max_harmonic_dependency_reduction_better
        )
        assert np.array_equal(expected_max_harmonics, harmonics)
