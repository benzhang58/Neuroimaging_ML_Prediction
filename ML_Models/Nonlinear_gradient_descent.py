import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def load_data(file_path, target_column_name):
    df = pd.read_csv(file_path)
    participant_ids = df.iloc[:, 0]
    y = df[target_column_name]
    x = df.drop(columns=[df.columns[0], target_column_name])
    return participant_ids, x, y

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient
        
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if i % 10000 == 0:
            print(f"Iteration {i}: Cost {cost}")
    
    return theta, cost_history

# Load the data
file_path = "cTDCS Data Pre.csv"
target_column_name = "AQ1"
participant_ids, x, y = load_data(file_path, target_column_name)

# Normalize features 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Impute NaN values in X and y
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(x_scaled)
y = np.nan_to_num(y, nan=np.nanmean(y))

# Check for NaN or infinite values
print("Checking for NaN or infinite values in X:")
print(np.isnan(X).any(), np.isinf(X).any())

print("Checking for NaN or infinite values in y:")
print(np.isnan(y).any(), np.isinf(y).any())

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Add an intercept term (a column of ones) to X_poly for the bias term
X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]

# Check the first few rows of X_poly for sanity check
print("First few rows of X_poly:")
print(X_poly[:5, :])

# Initialize theta (the parameters)
theta = np.random.randn(X_poly.shape[1]) * 0.01

# Set learning rate & number of iterations
learning_rate = 0.00001  # You may need to adjust this
num_iterations = 1000000  # You may need to adjust this

# Perform gradient descent
theta, cost_history = gradient_descent(X_poly, y, theta, learning_rate, num_iterations)

# Print final theta and cost
print("Final theta (parameters):")
print(theta)
print("Final cost:")
print(cost_history[-1])

# Calculate R-squared
y_pred = X_poly.dot(theta)
r2 = r2_score(y, y_pred)
print("R-squared:", r2)

# Plot cost history
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations')
plt.show()

# Plot actual vs predicted values
plt.scatter(y, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.show()
