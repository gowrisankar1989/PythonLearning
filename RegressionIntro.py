import quandl
import math
from matplotlib import style
import matplotlib.pyplot as plt
import time
import pandas as pd
from datetime import datetime as dt
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

#Learning  python by watching youtube series.

df = quandl.get('WIKI/GOOGL')
print (df.tail(n=5))
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))  # ,'adj_close'],1))

X = preprocessing.scale(X)
# print("X after preprocessing.scale ",X)
X_lately = X[-forecast_out:]
# print("X_lately",X_lately)
X = X[:-forecast_out]

# print(df)
print("X",X)
Y = np.array(df['label'])
# Y=preprocessing.scale(Y)
Y = Y[:-forecast_out]
print("Y ",Y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf = clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
forecast_set = clf.predict(X_lately)
# forecast_set_whole=clf.predict(X)
forecast_set = np.array(forecast_set)
df.dropna(inplace=True)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
print(last_date)
last_unix = time.mktime(time.strptime(str(last_date), "%Y-%m-%d %H:%M:%S"))  # .timestamp()
one_day = 86400
next_unix = last_unix + one_day
print(next_unix)

for i in forecast_set:
    next_date = dt.fromtimestamp(next_unix)
    print(next_date)
    next_unix += one_day
    print(next_unix)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    print(df.loc[next_date])
print (df)

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

