import pandas as pd
import numpy as np

# Function to load data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to compute the correlation matrix
def compute_correlation(data):
    return data.corr()

# Function to get the top correlated variables with the target variable
def top_correlated_variables(corr_matrix, target_variable, top_n=10):
    correlations = corr_matrix[target_variable].drop(target_variable)  # Exclude the target variable itself
    top_correlations = correlations.abs().nlargest(top_n)  # Get top_n correlations by absolute value
    return top_correlations.index.tolist(), top_correlations

def main():
    file_path = 'subcortical_imputed.csv'  # Replace with your actual CSV file path
    target_variable = 'stand_IQ'  # Target variable to correlate with
    
    # Load and preprocess the data
    data = load_data(file_path)
    data = data.select_dtypes(include=[np.number])  # Keep only numeric columns

    # Compute the correlation matrix
    corr_matrix = compute_correlation(data)

    # Get the top correlated variables with the target variable
    top_vars, top_correlations = top_correlated_variables(corr_matrix, target_variable)
    
    # Print the top correlated variables
    print(f"Top {len(top_vars)} variables most correlated with '{target_variable}':\n")
    print(top_correlations)

if __name__ == "__main__":
    main()
