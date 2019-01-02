from tbats import TBATS
import numpy as np

steps = 14
np.random.seed(2342)
t = np.array(range(0, 160))
y = 5 * np.sin(t * 2 * np.pi / 14.5) + 5 * np.cos(t * 2 * np.pi / 30.25) + \
    ((t / 20) ** 1.5 + np.random.normal(size=160) * t / 50) + 10
y = np.asarray(y)
y_to_train = y[:(len(y) - steps)]
y_to_predict = y[(len(y) - steps):]

estimator = TBATS(
    seasonal_periods=[14.5, 30.25],
    use_arma_errors=None,
    use_box_cox=False,
    use_trend=None,
    use_damped_trend=None,
    show_warnings=False,
)
fitted_model = estimator.fit(y_to_train)

# Warning messages from the model, if any
for warning in fitted_model.warnings:
    print(warning)

print('Did the model fit?', fitted_model.is_fitted)  # Model may fail to fit in edge-case situations
print('AIC', fitted_model.aic)  # may be np.inf

# Lets check components used in the model
print('\n\nMODEL SUMMARY\n\n')
params = fitted_model.params
components = fitted_model.params.components

print('Smoothing parameter', params.alpha)

print('Seasonal periods',
      components.seasonal_periods)  # TBATS may choose non-seasonal model even if you provide seasons
print('Harmonics amount for each season', components.seasonal_harmonics)
print('1st seasonal smoothing parameters', params.gamma_1())  # one value for each season
print('2nd seasonal smoothing parameters', params.gamma_2())  # one value for each season

print('Trend and damping', components.use_trend, components.use_damped_trend)
print('Trend', params.beta)
print('Damping', params.phi)

print('Use Box-Cox', components.use_box_cox)
print('Box-Cox lambda interval that was considered', components.box_cox_bounds)
print('Box-Cox lambda', params.box_cox_lambda)

print('ARMA residuals modelling', components.use_arma_errors)
print('ARMA(p, q)', components.p, components.q)
print('AR parameters', params.ar_coefs)
print('MA parameters', params.ma_coefs)

print('Seed state', params.x0)

# Short model summary
print('\n\nSUMMARY FUNCTION\n\n')
print(fitted_model.summary())

print('\n\nIN SAMPLE PREDICTIONS\n\n')
print('Original time series (5 first values)', fitted_model.y[:5])
print('Predictions (5 first values)', fitted_model.y_hat[:5])
print('Residuals (5 first values)', fitted_model.resid[:5])

y_forecasted = fitted_model.forecast(steps=steps)

print('\n\nFORECAST\n\n')
print('Values', y_forecasted)
print('MAE', np.mean(np.abs(y_forecasted - y_to_predict)))
