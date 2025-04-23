import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 100
n_steps = 50
noise1_std = 0.1  # Low noise (X̄₁)
noise2_std_range = np.linspace(2.0, 0.1, n_steps)  # High → Low noise (X̄₂)

# Storage
correlations = []
lasso_selected = []
elastic_selected = []

# Base true signal
X_true = np.random.uniform(-1, 1, n_samples)
Y = X_true  # True relationship

# Iterate over noise2 levels
for noise2_std in noise2_std_range:
    # Add noise
    X1 = X_true + np.random.normal(0, noise1_std, n_samples)
    X2 = X_true + np.random.normal(0, noise2_std, n_samples)

    # Stack features and standardize
    X = np.vstack((X1, X2)).T
    X_scaled = StandardScaler().fit_transform(X)

    # Compute correlation between features
    corr = np.corrcoef(X_scaled[:, 0], X_scaled[:, 1])[0, 1]
    correlations.append(corr)

    # LASSO
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_scaled, Y)
    lasso_selected.append(np.argmax(np.abs(lasso.coef_)))  # 0 or 1

    # Elastic Net
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
    enet.fit(X_scaled, Y)
    elastic_selected.append(np.argmax(np.abs(enet.coef_)))

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(correlations, lasso_selected, 'o-', label='LASSO Selected Feature')
plt.plot(correlations, elastic_selected, 's-', label='Elastic Net Selected Feature')
plt.axvline(0.7, color='gray', linestyle='--', label='Expected Threshold')
plt.xlabel("Correlation between $\\bar{X}_1$ and $\\bar{X}_2$")
plt.ylabel("Selected Feature (0 = $\\bar{X}_1$, 1 = $\\bar{X}_2$)")
plt.title("Feature Selection vs. Correlation")
plt.legend()
plt.grid(True)
plt.show()

# Previous unnecessary code

# from sklearn.linear_model import ElasticNet
# from sklearn.datasets import make_regression

# from pandas import read_csv
# from matplotlib import pyplot

# from numpy import mean
# from numpy import std
# from numpy import absolute
# from pandas import read_csv
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedKFold
# from sklearn.linear_model import ElasticNet

# X, y = make_regression(n_features=2, random_state=0)

# # load dataset
# url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
# dataframe = read_csv(url, header=None)

# # summarize shape
# print(dataframe.shape)
# # summarize first few lines
# print(dataframe.head())