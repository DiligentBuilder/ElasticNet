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

# Track when LASSO starts misbehaving
lasso_behavior_shift = []

for noise2_std in noise2_std_range:
    X1 = X_true + np.random.normal(0, noise1_std, n_samples)
    X2 = X_true + np.random.normal(0, noise2_std, n_samples)
    
    X = np.vstack((X1, X2)).T
    X_scaled = StandardScaler().fit_transform(X)
    
    # New dynamic Y depending on X2 noise
    w = 1.0 / (noise2_std + 1)
    w = w / (1 + w)  # normalize between 0 and 1
    Y = (1 - w) * X1 + w * X2

    corr = np.corrcoef(X_scaled[:, 0], X_scaled[:, 1])[0, 1]
    correlations.append(corr)

    # LASSO
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_scaled, Y)
    lasso_coefs.append(lasso.coef_)

    if np.all(lasso.coef_ == 0):
        lasso_selected.append(-1)
        lasso_behavior_shift.append(False)
    else:
        max_coefficient_index = np.argmax(np.abs(lasso.coef_))
        lasso_selected.append(max_coefficient_index)
        if np.count_nonzero(lasso.coef_) > 1:
            lasso_behavior_shift.append(True)
        else:
            lasso_behavior_shift.append(False)

    # Elastic Net
    enet = ElasticNet(alpha=0.01, l1_ratio=0.5)
    enet.fit(X_scaled, Y)
    elastic_coefs.append(enet.coef_)
    if np.all(enet.coef_ == 0):
        elastic_selected.append(-1)
    else:
        elastic_selected.append(np.argmax(np.abs(enet.coef_)))

import matplotlib.colors as mcolors

# Define a colormap
cmap = mcolors.ListedColormap(['gray', 'blue', 'green'])  # None = gray, X1 = blue, X2 = green

# Map selection to integers for plotting
# -1 -> 0 (gray for None), 0 -> 1 (blue for X1), 1 -> 2 (green for X2)
selection_map = { -1: 0, 0: 1, 1: 2 }
lasso_selection_mapped = [selection_map[sel] for sel in lasso_selected]

# Make a figure
plt.figure(figsize=(10, 2))

# Use scatter but fill the plot
plt.scatter(correlations, np.ones_like(correlations), 
            c=lasso_selection_mapped, cmap=cmap, marker='s', s=200)

# Formatting
plt.xlabel("Correlation between $\\bar{X}_1$ and $\\bar{X}_2$")
plt.title("LASSO Feature Selection Phase Diagram")
plt.yticks([])
plt.grid(False)
plt.xlim(min(correlations), max(correlations))
plt.tight_layout()
plt.show()



# Now, identify the correlation where LASSO's feature selection starts misbehaving (shifting)
lasso_outperformance_threshold = None
for i, corr in enumerate(correlations):
    if lasso_behavior_shift[i]:
        lasso_outperformance_threshold = corr
        break

# Plotting
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
if lasso_outperformance_threshold:
    plt.axvline(lasso_outperformance_threshold, color='red', linestyle='--', label="LASSO Outperformance Threshold")

# Labels and formatting
plt.xlabel("Correlation between $\\bar{X}_1$ and $\\bar{X}_2$")
plt.ylabel("Selected Feature\n(0 = $\\bar{X}_1$, 1 = $\\bar{X}_2$, -1 = None)")
plt.title("Feature Selection Behavior vs. Correlation (Jitter Added)")
plt.yticks([-1, 0, 1], ['None', '$\\bar{X}_1$', '$\\bar{X}_2$'])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print out the threshold value where LASSO starts misbehaving
if lasso_outperformance_threshold:
    print(f"LASSO starts misbehaving at a correlation of: {lasso_outperformance_threshold}")
else:
    print("No distinct behavior shift detected for LASSO.")
