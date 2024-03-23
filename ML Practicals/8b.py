
'''
8.b. Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs

pip instal matplotlib
'''
import numpy as np
import matplotlib.pyplot as plt

class LocallyWeightedRegression:
    def __init__(self, k):
        self.k = k  # Bandwidth parameter

    def gaussian_kernel(self, x, x_i):
        # Gaussian kernel function
        return np.exp(-np.sum((x - x_i) ** 2) / (2 * self.k ** 2))

    def fit(self, X, y, x_query):
        # Add a bias term to X
        X = np.c_[np.ones(X.shape[0]), X]
        x_query = np.r_[1, x_query]  # Add bias term to query point

        # Calculate weights for each data point
        weights = np.array([self.gaussian_kernel(x_query, x_i) for x_i in X])

        # Perform locally weighted linear regression
        W = np.diag(weights)
        theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

        # Predict the value for the query point
        y_pred = x_query @ theta

        return y_pred

# Example usage
# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, size=X.shape[0])

# Query point
x_query = 5

# Bandwidth parameter for Gaussian kernel
k = 1

# Fit the data points using Locally Weighted Regression
lwr = LocallyWeightedRegression(k)
y_pred = lwr.fit(X.reshape(-1, 1), y, x_query)

# Plot the original data points and the fitted regression curve
plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X, np.sin(X), color='green', label='True function')
plt.plot(x_query, y_pred, 'ro', label='Query point prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
