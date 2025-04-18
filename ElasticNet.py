from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression

from pandas import read_csv
from matplotlib import pyplot

from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet

X, y = make_regression(n_features=2, random_state=0)

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)

# summarize shape
print(dataframe.shape)
# summarize first few lines
print(dataframe.head())

from sklearn.linear_model import ElasticNet, ElasticNetCV

# Build the model
elastic_net_cv = ElasticNetCV(cv=5, random_state=1)

# Train the model
elastic_net_cv.fit(X_train, y_train)

print(f'Best Alpha: {elastic_net_cv.alpha_}')
print(f'Best L1 Ratio:{elastic_net_cv.l1_ratio_}')

from sklearn.metrics import mean_squared_error

# Predict values from the test dataset
elastic_net_pred = elastic_net_cv.predict(X_test)

mse = mean_squared_error(y_test, elastic_net_pred)
r_squared = elastic_net_cv.score(X_test, y_test)

print(f'Mean Squared Error: {mse}')
print(f'R-squared value: {r_squared}')
