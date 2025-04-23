import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 100
n_steps = 50
noise1_std = 0.1
noise2_std_range = np.linspace(2.0, 0.1, n_steps)

# Base signal and response
X_true = np.random.uniform(-1, 1, n_samples)
Y = X_true.copy()

# Storage
correlations = []
lasso_selected = []
elastic_selected = []
lasso_coefs = []
elastic_coefs = []

for noise2_std in noise2_std_range:
    X1 = X_true + np.random.normal(0, noise1_std, n_samples)
    X2 = X_true + np.random.normal(0, noise2_std, n_samples)

    X = np.vstack((X1, X2)).T
    X_scaled = StandardScaler().fit_transform(X)

    corr = np.corrcoef(X_scaled[:, 0], X_scaled[:, 1])[0, 1]
    correlations.append(corr)

    # LASSO (lower alpha)
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_scaled, Y)
    lasso_coefs.append(lasso.coef_)
    if np.all(lasso.coef_ == 0):
        lasso_selected.append(-1)  # Nothing selected
    else:
        lasso_selected.append(np.argmax(np.abs(lasso.coef_)))

    # Elastic Net (also lower alpha)
    enet = ElasticNet(alpha=0.01, l1_ratio=0.5)
    enet.fit(X_scaled, Y)
    elastic_coefs.append(enet.coef_)
    if np.all(enet.coef_ == 0):
        elastic_selected.append(-1)
    else:
        elastic_selected.append(np.argmax(np.abs(enet.coef_)))

# Plotting
# Example usage in your simulation plotting:
plt.figure(figsize=(10, 6))

# Create jitter for better separation
jitter_strength = 0.05
lasso_jitter_y = np.random.uniform(-jitter_strength, jitter_strength, len(lasso_selected))
elastic_jitter_y = np.random.uniform(-jitter_strength, jitter_strength, len(elastic_selected))

# Slight x-offset to avoid exact overlap
lasso_x = np.array(correlations) - 0.005
elastic_x = np.array(correlations) + 0.005

# Scatter with transparency (alpha) and size
plt.scatter(lasso_x, np.array(lasso_selected) + lasso_jitter_y,
            label="LASSO", alpha=0.7, color='blue', marker='o', s=80)

plt.scatter(elastic_x, np.array(elastic_selected) + elastic_jitter_y,
            label="Elastic Net", alpha=0.7, color='green', marker='s', s=80)

# Add threshold line
plt.axvline(0.7, color='gray', linestyle='--', label="Expected Threshold")

# Labels and formatting
plt.xlabel("Correlation between $\\bar{X}_1$ and $\\bar{X}_2$")
plt.ylabel("Selected Feature\n(0 = $\\bar{X}_1$, 1 = $\\bar{X}_2$, -1 = None)")
plt.title("Feature Selection Behavior vs. Correlation")
plt.yticks([-1, 0, 1], ['None', '$\\bar{X}_1$', '$\\bar{X}_2$'])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()