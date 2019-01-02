from tbats import TBATS
import numpy as np

steps = 14
np.random.seed(2342)
y = 0.2 * np.array(range(0, 100)) + np.random.normal(size=100)

estimator = TBATS()

# 100 observations
fitted_model = estimator.fit(y)

forecast = fitted_model.forecast(steps=1)

print('Forecast for day 101 is', forecast[0])

# New observation is available
new_observation = 20.7
print('Forecast error', new_observation - forecast[0])

y = np.append(y, new_observation)

fitted_model.fit(y)  # Re-calculate model for new observation, it will not change model parameters
forecast = fitted_model.forecast(steps=1)

print('Forecast for day 102 is', forecast[0])
