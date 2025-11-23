import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os

np.random.seed(20)
n_samples = 100
X = np.random.uniform(-1, 1, size=(n_samples, 1))
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(scale=1, size=n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

degrees = np.arange(1, 130)
train_errors = []
test_errors = []

os.makedirs("figures", exist_ok=True)

for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = Ridge(alpha=0)
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(degrees[0], degrees[-1])
ax.set_yscale('log')
ax.set_ylim(min(min(train_errors), min(test_errors))*0.8, max(max(train_errors), max(test_errors))*1.2)
ax.set_xlabel("Polynomial Degree")
ax.set_ylabel("Mean Squared Error (log scale)")
ax.set_title("Double Descent in Polynomial Regression")
ax.grid(True, linestyle='--', alpha=0.5)

line_train, = ax.plot([], [], 'o-', color='dodgerblue', label='Train MSE', lw=2, alpha=0.8)
line_test, = ax.plot([], [], 'o-', color='orange', label='Test MSE', lw=2, alpha=0.8)
ax.legend()

def init():
    line_train.set_data([], [])
    line_test.set_data([], [])
    return line_train, line_test

def update(frame):
    line_train.set_data(degrees[:frame], train_errors[:frame])
    line_test.set_data(degrees[:frame], test_errors[:frame])
    return line_train, line_test

ani = FuncAnimation(fig, update, frames=len(degrees), init_func=init,
                    blit=True, interval=30)

ani.save("figures/double_descent_animation.gif", writer='pillow', dpi=150)
plt.close()
