from tbats import TBATS
import numpy as np

np.random.seed(2342)
t = np.array(range(0, 160))
y = 5 * np.sin(t * 2 * np.pi / 7) + 2 * np.cos(t * 2 * np.pi / 30.5) + \
    ((t / 20) ** 1.5 + np.random.normal(size=160) * t / 50) + 10

# Create estimator
estimator = TBATS(seasonal_periods=[14, 30.5])

# Fit model
fitted_model = estimator.fit(y)

# Forecast 14 steps ahead
y_forecasted = fitted_model.forecast(steps=14)

# Summarize fitted model
print(fitted_model.summary())
