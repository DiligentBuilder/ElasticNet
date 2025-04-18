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

# Method 1

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



# Method 2

import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('heart_disease.csv')  # Replace with your dataset
X = data.drop('target', axis=1)  # Features
y = data['target']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create a pipeline to scale features and apply Elastic Net Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('model', LogisticRegression(
        penalty='elasticnet',
        solver='saga',  # Required for Elastic Net
        l1_ratio=0.5,  # Mix of L1 and L2
        max_iter=1000  # Increase iterations for convergence
    ))
])

# Fit the model
pipeline.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, roc_auc_score

# Make predictions
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_prob))

from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__l1_ratio': [0.1, 0.5, 0.7, 0.9]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Get feature coefficients
coefficients = grid_search.best_estimator_.named_steps['model'].coef_[0]
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(feature_importance.head())
