from tbats import BATS
import numpy as np

steps = 14
np.random.seed(2342)
t = np.array(range(0, 160))
y = 5 * np.sin(t * 2 * np.pi / 14) + \
    ((t / 20) ** 1.5 + np.random.normal(size=160) * t / 50) + 10
y = np.asarray(y)
y_to_train = y[:(len(y) - steps)]
y_to_predict = y[(len(y) - steps):]

estimator = BATS(
    seasonal_periods=[14],
    use_arma_errors=True,
    use_box_cox=False,
)
fitted_model = estimator.fit(y_to_train)
y_forecasted = fitted_model.forecast(steps=steps)

print('MAE', np.mean(np.abs(y_forecasted - y_to_predict)))

# Short model summary
print(fitted_model.summary())
