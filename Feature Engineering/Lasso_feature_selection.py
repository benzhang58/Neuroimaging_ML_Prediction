import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Function to load data from a CSV file
def load_data(file_path, target_column):
    data = pd.read_csv(file_path)
    data = data.drop(columns=[data.columns[0]])  # Drop the subject identifier column

    X = data.drop(columns=[target_column])  # Features
    y = data[target_column]  # Target variable
    
    # Handle empty cells by filling them with the mean of the column
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    return X, y

# Function to perform Lasso feature selection
def lasso_feature_selection(X, y, alpha=0.01, max_iter=1000000):
    # Standardize the feature matrix
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Lasso regression with more iterations
    lasso = Lasso(alpha=alpha, max_iter=max_iter)
    lasso.fit(X_scaled, y)
    
    # Select features with non-zero coefficients
    selected_features = X.columns[lasso.coef_ != 0]
    return selected_features, lasso.coef_[lasso.coef_ != 0]

# Main function to load data and perform feature selection
def main():
    file_path = 'All_features.csv'  # Replace with your actual CSV file path
    target_column = 'raw_IQ'  # Replace with your actual target variable name
    X, y = load_data(file_path, target_column)

    # Perform Lasso feature selection
    selected_features, coefficients = lasso_feature_selection(X, y, alpha=0.01, max_iter=10000)

    # Display the selected features and their coefficients
    print(f"Selected features: {selected_features}")
    print(f"Coefficients: {coefficients}")

    # Optionally, evaluate the selected features with cross-validation
    X_selected = X[selected_features]
    scores = cross_val_score(Lasso(alpha=0.01), X_selected, y, cv=5, scoring='r2')
    print(f"R-squared scores: {scores}")
    print(f"Average R-squared: {scores.mean()}")

if __name__ == "__main__":
    main()
