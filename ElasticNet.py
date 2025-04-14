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