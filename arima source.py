from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split, SlidingWindowSplitter, ForecastingGridSearchCV
from sktime.utils.plotting import plot_series
from sktime.forecasting.naive import NaiveForecaster
from sklearn.metrics import mean_absolute_percentage_error
from sktime.forecasting.arima import ARIMA, AutoARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore")

# setting graphs size
plt.rcParams["figure.figsize"] = [16,7]
# for fancy plots
plt.style.use('ggplot')

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv',
                 parse_dates=['date'], index_col="date")

df.index = pd.PeriodIndex(df.index, freq="M")

series = df.T.iloc[0]

plot_series(series);


model_auto = AutoARIMA(sp=12, suppress_warnings=True).fit(series)

summary = model_auto.summary()

def get_params(summary_text):
  full = re.findall(r'SARIMAX\(.*?\)x\(.*?\)', summary_text)[0]
  info = [int(_) for _ in re.findall(r'\d+', full)]
  return info


p, d, q, P, D, Q, S = get_params(summary.as_text())

y_train, y_test = temporal_train_test_split(series, test_size=24)



fh = ForecastingHorizon(y_test.index, is_relative=False)


plot_series(y_train, y_test, labels=['Train', 'Test']);



model = NaiveForecaster(strategy="last", sp=12).fit(y_train)



y_pred = model.predict(fh)


print(type(y_pred))
print(y_pred)


plot_series(y_train, y_test, y_pred, labels=['Train', 'Test', 'Predicted']);


plot_series(y_test, y_pred, labels=['Test', 'Predicted']);

mape_naive = mean_absolute_percentage_error(y_pred, y_test)

model = ARIMA(order = (p, d, q)).fit(y_train)

y_pred, y_conf = model.predict(fh, return_pred_int=True)

y_train.plot(label='Train')
y_test.plot(label='Test')
y_pred.plot(label='Predicted')

plt.fill_between(y_conf.index, y_conf.lower, y_conf.upper, color= 'magenta', alpha=.3)

plt.scatter(y_train.index, y_train)
plt.scatter(y_test.index, y_test)
plt.scatter(y_pred.index, y_pred)

plt.legend()
plt.show()

y_test.plot(label='Test')
y_pred.plot(label='Predicted')

plt.fill_between(y_conf.index, y_conf.lower, y_conf.upper, color= 'magenta', alpha= .3)

plt.scatter(y_test.index, y_test)
plt.scatter(y_pred.index, y_pred)

plt.legend()
plt.show()


mape_arima = mean_absolute_percentage_error(y_pred, y_test)

model = ARIMA(order = (p, d, q), seasonal_order = (P, D, Q, S)).fit(y_train)

y_pred, y_conf = model.predict(fh, return_pred_int=True)

y_train.plot(label='Train')
y_test.plot(label='Test')
y_pred.plot(label='Predicted')

plt.fill_between(y_conf.index, y_conf.lower, y_conf.upper, color= 'magenta', alpha=.3)

plt.scatter(y_train.index, y_train)
plt.scatter(y_test.index, y_test)
plt.scatter(y_pred.index, y_pred)

plt.legend()
plt.show()


y_test.plot(label='Test')
y_pred.plot(label='Predicted')

plt.fill_between(y_conf.index, y_conf.lower, y_conf.upper, color= 'magenta', alpha= .3)

plt.scatter(y_test.index, y_test)
plt.scatter(y_pred.index, y_pred)

plt.legend()
plt.show()


mape_arima_seas = mean_absolute_percentage_error(y_pred, y_test)
mape_arima_seas


print("Naive Model \t \t", mape_naive)
print("Arima no Seas.\t \t", mape_arima)
print("Arima with Seas.\t", mape_arima_seas)

series_log = series.apply(np.log)

plot_series(series_log);


fh = ForecastingHorizon(series[-25:].index, is_relative=False)


y_train, y_test = temporal_train_test_split(series.iloc[train_[0]:test_[0]+1], test_size=1)

y_predA, y_confA = ForecastingGridSearchCV(ARIMA(), SlidingWindowSplitter(window_length=48, start_with_window=True, initial_window=48),
                              {'order': [(p, d, q)]}, n_jobs=-1).fit(series).predict(fh, return_pred_int=True)

y_predAS, y_confAS = ForecastingGridSearchCV(ARIMA(), SlidingWindowSplitter(window_length=48, start_with_window=True, initial_window=48),
                              {'order': [(p, d, q)], 'seasonal_order': [(P, D, Q, S)]}, n_jobs=-1).fit(series).predict(fh, return_pred_int=True)

y_predN = ForecastingGridSearchCV(NaiveForecaster(), SlidingWindowSplitter(window_length=48, start_with_window=True, initial_window=48),
                              {'strategy': ["last"], 'sp': [12]}, n_jobs=-1).fit(series).predict(fh)


errors_arima = (mean_absolute_percentage_error(series[-25:], y_predA))
errors_arima_seas = (mean_absolute_percentage_error(series[-25:], y_predAS))
errors_naive = (mean_absolute_percentage_error(series[-25:], y_predN))

print("Naive Model \t \t", np.mean(errors_naive))
print("Arima no Seas.\t \t", np.mean(errors_arima))
print("Arima with Seas.\t", np.mean(errors_arima_seas))

for y_pred, y_conf, title in [(y_predA, y_confA, 'Arima without seasionality'),
                       (y_predAS, y_confAS, 'Arima with seasionality'), (y_predN, False, 'Naive seasonal')]:
    plt.figure(figsize=(16,4))
    plt.title(title)
    series[:-25].plot(label='Train')
    series[-25:].plot(label='Test')
    y_pred.plot(label='Predicted')

    if type(y_conf) != bool:
      plt.fill_between(y_conf.index, y_conf.lower, y_conf.upper, color= 'magenta', alpha=.3)

    plt.scatter(series[:-25].index, series[:-25])
    plt.scatter(series[-25:].index, series[-25:])
    plt.scatter(y_pred.index, y_pred)

    plt.legend()
    plt.show()
    print("-"*150)
    plt.figure(figsize=(16,4))
    series[-25:].plot(label='Test')
    y_pred.plot(label='Predicted')

    if type(y_conf) != bool:
      plt.fill_between(y_conf.index, y_conf.lower, y_conf.upper, color= 'magenta', alpha=.3)

    plt.scatter(series[-25:].index, series[-25:])
    plt.scatter(y_pred.index, y_pred)
   
    plt.legend()
    plt.show()





