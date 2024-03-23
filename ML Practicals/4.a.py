'''
4a For a given set of training data examples stored in a .CSV file implement Least Square Regression algorithm.
'''

import pandas as pd
import numpy as np

# Load data from CSV
data = pd.read_csv('train.csv')

# Convert data to numeric format
data_numeric = data.apply(pd.to_numeric, errors='coerce')
data_numeric.dropna(inplace=True)  # Drop rows with missing values

# Separate features (X) and target variable (y)
X = data_numeric.iloc[:, :-1].values
y = data_numeric.iloc[:, -1].values

# Add bias term (intercept) to X
X = np.c_[np.ones(X.shape[0]), X]

# Convert y to a numpy array
y = np.array(y)

# Compute the coefficients using pseudoinverse
coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y

# Print the coefficients
print("Coefficients:", coefficients)
