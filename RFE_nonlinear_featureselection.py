import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

# Function to perform RFE with a DecisionTreeRegressor
def rfe_with_decision_tree(X, y, n_features_to_select):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize the DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=42)
    
    # Perform RFE
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X_scaled, y)
    
    # Get the selected features
    selected_features = X.columns[rfe.support_]
    return selected_features, rfe

# Main function to run RFE and then use selected features
def main():
    file_path = 'All_features.csv'  # Replace with your actual CSV file path
    target_column = 'raw_IQ'  # Replace with your actual target variable name
    X, y = load_data(file_path, target_column)

    # Perform RFE to select top features
    n_features_to_select = 20  # Set the number of features you want to select
    selected_features, rfe = rfe_with_decision_tree(X, y, n_features_to_select)
    
    print(f"Selected features: {selected_features}")
    
    # Optionally, you can now use these selected features to train your neural network
    # X_selected = X[selected_features]
    # Train your neural network with X_selected and y

if __name__ == "__main__":
    main()
