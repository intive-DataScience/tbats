from tbats import TBATS
import numpy as np

# required on windows for multi-processing,
# see https://docs.python.org/2/library/multiprocessing.html#windows
if __name__ == '__main__':
    alpha = 0.4
    beta = 0.6
    phi = 0.9
    np.random.seed(987)
    T = 120

    b = 0
    l = 1
    y = [0] * T
    for t in range(0, T):
        d = np.random.normal(scale=1.0)
        y[t] = l + b + d
        l = l + b + alpha * d
        b = phi * b + beta * d

    y_to_train = y[:100]
    y_to_forecast = y[100:]

    # Create estimator
    estimator = TBATS(use_box_cox=False, use_arma_errors=False)

    # Fit model
    fitted_model = estimator.fit(y_to_train)

    # Forecast 14 steps ahead, also get confidence intervals for 95% level
    # When you provide level it returns a tuple
    y_forecasted, confidence_info = fitted_model.forecast(steps=20, confidence_level=0.95)

    # Summarize fitted model
    print(fitted_model.summary())

    # Print mean absolute errors
    print('MAE (in sample)', np.mean(np.abs(fitted_model.resid)))
    print('MAE (forecast)', np.mean(np.abs(y_forecasted - y_to_forecast)))

    # Print forecast confidence intervals
    print('Calculated for confidence level:', confidence_info['calculated_for_level'])

    print('CALCULATIONS ARE VALID ONLY WHEN RESIDUALS HAVE NORMAL DISTRIBUTION')

    print('Lower bound:', confidence_info['lower_bound'])
    print('Predictions:', confidence_info['mean'])
    print('Upper bound:', confidence_info['upper_bound'])
