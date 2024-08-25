import numpy as np  # Import the numpy library for numerical computations
import pandas as pd  # Import the pandas library for data manipulation
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for data normalization
from sklearn.model_selection import KFold  # Import KFold for cross-validation
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.metrics import r2_score


# Function to load and prepare data from a CSV file
def load_data(file_path, target_column):
    data = pd.read_csv(file_path)  # Load the CSV file into a pandas DataFrame
    data = data.drop(columns=[data.columns[0]])  # Drop the subject identifier column
    
    # Handle empty cells by filling them with the mean of the column
    data = data.fillna(data.mean())
    
    X = data.drop(columns=[target_column]).values  # Extract features
    y = data[target_column].values  # Extract the target variable
    
    scaler = StandardScaler()  # Initialize the StandardScaler
    X = scaler.fit_transform(X)  # Standardize the input data
    
    return X, y  #

# Function to initialize weights and biases for the neural network
def initialize_parameters(input_size, hidden_size1, hidden_size2, output_size):
    W1 = np.random.randn(hidden_size1, input_size) * 0.01  # Initialize weights for the first layer (input to hidden1)
    b1 = np.zeros((hidden_size1, 1))  # Initialize biases for the first layer
    
    W2 = np.random.randn(hidden_size2, hidden_size1) * 0.01  # Initialize weights for the second layer (hidden1 to hidden2)
    b2 = np.zeros((hidden_size2, 1))  # Initialize biases for the second layer
    
    W3 = np.random.randn(output_size, hidden_size2) * 0.01  # Initialize weights for the third layer (hidden2 to output)
    b3 = np.zeros((output_size, 1))  # Initialize biases for the third layer
    
    return W1, b1, W2, b2, W3, b3  # Return the initialized weights and biases

# Function to implement the Leaky ReLU activation function
def leaky_relu(Z):
    return np.maximum(0.01 * Z, Z)  # Apply the Leaky ReLU activation function

# Function to calculate the derivative of the Leaky ReLU function
def leaky_relu_derivative(Z):
    return np.where(Z > 0, 1, 0.01)  # Compute the derivative: 1 if Z > 0, else 0.01

# Function to perform forward propagation through the network
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(W1, X) + b1  # Compute the linear transformation for the first layer
    A1 = leaky_relu(Z1)  # Apply the Leaky ReLU activation function
    
    Z2 = np.dot(W2, A1) + b2  # Compute the linear transformation for the second layer
    A2 = leaky_relu(Z2)  # Apply the Leaky ReLU activation function
    
    Z3 = np.dot(W3, A2) + b3  # Compute the linear transformation for the output layer
    A3 = Z3  # No activation function for the output layer (linear output)
    
    return Z1, A1, Z2, A2, Z3, A3  # Return the computed values for each layer

# Function to compute the cost (mean squared error) of the predictions with L2 regularization
def compute_cost(A3, y, W1, W2, W3, lambda_):
    m = y.shape[1]  # Get the number of examples
    cost = (1 / (2 * m)) * np.sum(np.square(A3 - y))  # Compute the mean squared error
    
    # Add L2 regularization term
    l2_regularization_cost = (lambda_ / (2 * m)) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    
    total_cost = cost + l2_regularization_cost  # Total cost with L2 regularization
    return total_cost  # Return the computed cost

# Function to perform backward propagation to compute gradients
def backward_propagation(X, y, Z1, A1, Z2, A2, Z3, A3, W2, W3):
    m = y.shape[1]  # Get the number of examples
    
    dZ3 = A3 - y  # Compute the gradient of the cost with respect to Z3 (output layer)
    dW3 = (1 / m) * np.dot(dZ3, A2.T)  # Compute the gradient of the cost with respect to W3
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)  # Compute the gradient of the cost with respect to b3
    
    dA2 = np.dot(W3.T, dZ3)  # Compute the gradient of the cost with respect to A2
    dZ2 = dA2 * leaky_relu_derivative(Z2)  # Compute the gradient of the cost with respect to Z2
    dW2 = (1 / m) * np.dot(dZ2, A1.T)  # Compute the gradient of the cost with respect to W2
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # Compute the gradient of the cost with respect to b2
    
    dA1 = np.dot(W2.T, dZ2)  # Compute the gradient of the cost with respect to A1
    dZ1 = dA1 * leaky_relu_derivative(Z1)  # Compute the gradient of the cost with respect to Z1
    dW1 = (1 / m) * np.dot(dZ1, X.T)  # Compute the gradient of the cost with respect to W1
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # Compute the gradient of the cost with respect to b1
    
    return dW1, db1, dW2, db2, dW3, db3  # Return the computed gradients

# Function to update the weights and biases using gradient descent
def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 = W1 - learning_rate * dW1  # Update weights for the first layer
    b1 = b1 - learning_rate * db1  # Update biases for the first layer
    
    W2 = W2 - learning_rate * dW2  # Update weights for the second layer
    b2 = b2 - learning_rate * db2  # Update biases for the second layer
    
    W3 = W3 - learning_rate * dW3  # Update weights for the third layer
    b3 = b3 - learning_rate * db3  # Update biases for the third layer
    
    return W1, b1, W2, b2, W3, b3  # Return the updated weights and biases

# Function to train the neural network
def train_neural_network(X, y, hidden_size1, hidden_size2, num_iterations, learning_rate, lambda_):
    input_size = X.shape[0]  # Get the size of the input layer
    output_size = 1  # Set the size of the output layer (for regression)
    W1, b1, W2, b2, W3, b3 = initialize_parameters(input_size, hidden_size1, hidden_size2, output_size)  # Initialize weights and biases

    costs = []  # Initialize a list to store the cost at each iteration

    for i in range(num_iterations):  # Loop over the number of iterations
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)  # Perform forward propagation
        cost = compute_cost(A3, y, W1, W2, W3, lambda_)  # Compute the cost with L2 regularization
        costs.append(cost)  # Store the cost in the list
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(X, y, Z1, A1, Z2, A2, Z3, A3, W2, W3)  # Perform backward propagation
        W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)  # Update parameters

        if i % 100 == 0:  # Print the cost every 100 iterations
            print(f"Iteration {i}, Cost: {cost}")

    return W1, b1, W2, b2, W3, b3, costs  # Return the trained parameters and the cost history

# Function to predict outcomes using the trained neural network
def predict(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)  # Perform forward propagation to get predictions
    return A3  # Return the predictions

# Main function to run the neural network regression with k-fold cross-validation
def main():
    file_path = 'feature_selected_features.csv'  # Path to the CSV file containing the data
    target_column = 'raw_IQ'  # Name of the target variable in the CSV file
    X, y = load_data(file_path, target_column)  # Load the data from the CSV file

    X = X.T  # Transpose X to match the input shape required by the neural network
    y = y.reshape(1, -1)  # Reshape y to match the output shape

    hidden_size1 = 10  # Number of neurons in the first hidden layer
    hidden_size2 = 5 # Number of neurons in the second hidden layer
    num_iterations = 30000  # Number of iterations for gradient descent
    learning_rate = 0.0001  # Learning rate for gradient descent
    lambda_ = 0.1  # Regularization parameter for L2 regularization

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_costs = []  # List to store the cost for each fold
    r_squared_values = []  # List to store R-squared values for each fold

    plt.figure(figsize=(10, 6))  # Initialize the plot for learning curves

    for fold, (train_index, test_index) in enumerate(kf.split(X.T), 1):  # Split the data into k folds
        X_train, X_test = X[:, train_index], X[:, test_index]  # Training and testing data
        y_train, y_test = y[:, train_index], y[:, test_index]  # Training and testing labels

        # Train the neural network on the training data
        W1, b1, W2, b2, W3, b3, costs = train_neural_network(X_train, y_train, hidden_size1, hidden_size2, num_iterations, learning_rate, lambda_)

        # Predict outcomes on the testing data
        predictions = predict(X_test, W1, b1, W2, b2, W3, b3)

        # Compute the cost for this fold and append it to fold_costs
        cost = compute_cost(predictions, y_test, W1, W2, W3, lambda_)
        fold_costs.append(cost)

        # Compute R-squared for this fold and append it to r_squared_values
        r_squared = r2_score(y_test.flatten(), predictions.flatten())
        r_squared_values.append(r_squared)

        # Plot the learning curve for this fold
        plt.plot(costs, label=f'Fold {fold} Cost')

    # Final plot adjustments
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Learning Curves for Each Fold')
    plt.legend()
    plt.show()

    # Train the neural network on the full dataset
    W1, b1, W2, b2, W3, b3, costs = train_neural_network(X, y, hidden_size1, hidden_size2, num_iterations, learning_rate, lambda_)

    # Predict outcomes using the trained neural network on the full dataset
    predictions = predict(X, W1, b1, W2, b2, W3, b3)
    print("Predictions:", predictions)  # Print the predictions

    print("Final Cost:", costs[-1])  # Print the final cost after training

    # Print the cost and R-squared for each fold and the averages
    print("Fold costs:", fold_costs)
    print("Average cost:", np.mean(fold_costs))
    print("R-squared values:", r_squared_values)
    print("Average R-squared:", np.mean(r_squared_values))

    # Optionally plot the cost over iterations for the full dataset
    plt.figure(figsize=(10, 6))
    plt.plot(costs, label='Full Training Set Cost')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Over Time (Full Dataset)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()  # Run the main function
