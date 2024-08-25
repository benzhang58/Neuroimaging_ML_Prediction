import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def load_data(file_path, target_column_name):
    # Load the data into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Extract the participant IDs (assuming it's the first column)
    participant_ids = df.iloc[:, 0]
    
    # Extract the target variable (y)
    y = df[target_column_name]
    
    # Extract the features (x) by dropping the participant ID and target column
    x = df.drop(columns=[df.columns[0], target_column_name])
    
    return participant_ids, x, y

def compute_cost(X, y, theta):
    """
    Compute the cost function for linear regression.
    
    X: Matrix of input features (m x n)
    y: Vector of target values (m x 1)
    theta: Vector of parameters/weights (n x 1)
    
    Returns the cost value.
    """
    m = len(y)  # Number of training examples
    predictions = X.dot(theta)  # Predicted values (hypothesis)
    errors = predictions - y  # Difference between predicted and actual values
    cost = (1 / (2 * m)) * np.sum(errors ** 2)  # Mean Squared Error
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    """
    Perform gradient descent to learn theta.
    
    X: Matrix of input features (m x n)
    y: Vector of target values (m x 1)
    theta: Initial vector of parameters/weights (n x 1)
    learning_rate: The step size for each iteration of gradient descent
    num_iterations: Number of iterations to perform
    
    Returns the final theta and a history of the cost function values.
    """
    m = len(y)  # Number of training examples
    cost_history = []  # To store the cost value after each iteration

    for i in range(num_iterations):
        predictions = X.dot(theta)  # Predicted values for the current theta
        errors = predictions - y  # Errors in prediction
        gradient = (1 / m) * X.T.dot(errors)  # Gradient of the cost function with respect to theta
        theta = theta - learning_rate * gradient  # Update theta by moving in the direction of the negative gradient
        
        cost = compute_cost(X, y, theta)  # Calculate the cost for the updated theta
        cost_history.append(cost)  # Save the cost in history for later analysis

        # Optional: print the cost every 100 iterations
        if i % 100 == 0:
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

# Add an intercept term (a column of ones) to X for the bias term in the linear model
X = np.c_[np.ones(x_scaled.shape[0]), x_scaled]  # Adding a column of ones to X

# Check the first few rows of X for sanity check
print("First few rows of X:")
print(X[:5, :])

# Initialize theta (the parameters)
theta = np.zeros(X.shape[1]) 

# Set learning rate & number of iterations
learning_rate = 0.001  # You can adjust this value
num_iterations = 10000  # Number of iterations for gradient descent

# Perform gradient descent
theta, cost_history = gradient_descent(X, y, theta, learning_rate, num_iterations)

# Print final theta and cost
print("Final theta (parameters):")
print(theta)
print("Final cost:")
print(cost_history[-1])

# Calculate R-squared
y_pred = X.dot(theta)
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
