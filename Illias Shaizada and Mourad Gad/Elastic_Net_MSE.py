import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
np.random.seed(42)
n_samples = 100  # Increased dataset size for reliability
X = np.random.uniform(-1, 1, size=(n_samples, 1))
y = np.sin(2 * np.pi * X).ravel() + 1 * np.random.normal(size=n_samples)  # Reduced noise

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

# Initialize arrays for results
train_errors = []
test_errors = []
degrees = np.arange(1, 200)  # Model complexity: Polynomial degrees

# Iterate over polynomial degrees
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Fit polynomial regression model with regularization
    model = ElasticNet(alpha=0.1, l1_ratio=0.2)
    model.fit(X_train_poly, y_train)

    # Calculate train and test errors
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, label='Train Loss', marker='o')
plt.plot(degrees, test_errors, label='Test Loss', marker='o')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Mean Squared Error')
plt.title('Double Descent in Polynomial Regression (Improved Setup)')
plt.legend()
plt.grid()
plt.show()