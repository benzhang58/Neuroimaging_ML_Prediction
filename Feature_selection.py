import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Function to load data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to calculate VIF for all features in a DataFrame
def calculate_vif(X):
    X = X.replace(["", np.inf, -np.inf], np.nan).dropna()  # Replace empty strings and inf with NaN, then drop those rows

    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

# Function to select features based on correlation and VIF
def select_features(data, target_column, correlation_threshold=0.1, vif_threshold=15):
    # Extract target variable and features
    y = data[target_column].replace("", np.nan)  # Replace empty strings with NaN
    X = data.drop(columns=[target_column])

    # Convert all values to numeric, forcing errors to NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Replace inf values with NaN and drop rows with NaN values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_data = pd.concat([X, y], axis=1).dropna()
    X = combined_data.drop(columns=[target_column])
    y = combined_data[target_column]

    # Calculate correlation with the target variable
    correlations = X.corrwith(y).abs()
    
    # Select features with correlation higher than the threshold
    selected_features = correlations[correlations > correlation_threshold].index.tolist()
    X_selected = X[selected_features]
    
    # Calculate VIF and remove features with high multicollinearity
    while True:
        vif = calculate_vif(X_selected)
        max_vif = vif["VIF"].max()

        if max_vif > vif_threshold:
            # Drop the feature with the highest VIF
            drop_feature = vif.loc[vif['VIF'].idxmax(), 'variables']
            X_selected = X_selected.drop(columns=[drop_feature])
        else:
            break
    
    return X_selected.columns.tolist(), X_selected, y

# Function to perform Recursive Feature Elimination (RFE) based on selected features
def perform_rfe(X, y, num_features=10):
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=num_features)
    rfe = rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return selected_features

# Main function to run the feature selection process
def main():
    file_path = 'All_features.csv'  # Replace with your actual CSV file path
    target_column = 'stand_IQ'  # Replace with your actual target variable name
    data = load_data(file_path)

    # Select features based on correlation and VIF
    selected_features, X_final, y_final = select_features(data, target_column)

    # Ensure no empty values in y_final before RFE
    X_final = X_final[~y_final.isna()]
    y_final = y_final.dropna()

    # Perform RFE to further refine feature selection
    final_features = perform_rfe(X_final, y_final, num_features=10)
    
    # Display the final selected features
    print("Final selected features:", final_features)
    
    # Display the correlation of the final selected features with the target variable
    correlations_final = X_final[final_features].corrwith(y_final)
    print("\nCorrelation of final selected features with the target variable:")
    print(correlations_final)

if __name__ == "__main__":
    main()
