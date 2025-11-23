import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
np.random.seed(20)
n_samples = 100  # Increased dataset size for reliability
X = np.random.uniform(-1, 1, size=(n_samples, 1))
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(scale=1, size=n_samples)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

poly = PolynomialFeatures(129) # Set polynomial degree
X_train_poly = poly.fit_transform(X_train)
X_curve = np.linspace(-1, 1, 100).reshape(-1, 1)
X_curve_poly = poly.transform(X_curve)

l1_ratios = [0.2, 0.4, 0.5, 0.7, 0.9]
colors = ['r', 'g', 'y', 'c', 'm']

for i, l1_ratio in enumerate(l1_ratios): # Plotting Elastic Net regression with different l1_ratios
  reg = ElasticNet(alpha=0.02, l1_ratio=l1_ratio)
  reg.fit(X_train_poly, y_train)
  y_curve = reg.predict(X_curve_poly)
  plt.plot(X_curve, y_curve, color=colors[i], label=f'l1_ratio={l1_ratio}')

# Plot curves
plt.scatter(X_test, y_test)
plt.title('Polynomial Regression of Degree 129 with Elastic Net')
plt.grid()
plt.legend()
plt.show()