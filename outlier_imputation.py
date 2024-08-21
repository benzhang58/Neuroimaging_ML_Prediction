import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats

# Function to detect outliers using Z-score
def detect_outliers(df, threshold=3):
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    z_scores = np.abs(stats.zscore(numeric_df))
    outliers = (z_scores > threshold)
    df_no_outliers = df.copy()
    df_no_outliers[outliers] = np.nan
    return df_no_outliers, outliers

# Function to perform KNN imputation and track imputed values
def impute_knn(df, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed_data, columns=df.columns)

    imputed_values = df_imputed.where(pd.isna(df), np.nan)  # Keep track of what was imputed
    return df_imputed, imputed_values

def main(input_csv, output_csv, n_neighbors=5):
    df = pd.read_csv(input_csv)

    # Remove the first column (assumed to be non-numeric participant IDs)
    participant_ids = df.iloc[:, 0]
    df_numeric = df.iloc[:, 1:]

    # Detect and replace outliers with NaN
    df_no_outliers, outliers = detect_outliers(df_numeric)

    # Impute missing values using KNN
    df_imputed, imputed_values = impute_knn(df_no_outliers, n_neighbors)

    # Add participant IDs back to the DataFrame
    df_imputed.insert(0, df.columns[0], participant_ids)

    # Save the imputed DataFrame to CSV
    df_imputed.to_csv(output_csv, index=False)

    # Save the DataFrame showing imputed values to a separate CSV
    imputed_values_output = output_csv.replace(".csv", "_imputed_values.csv")
    imputed_values.insert(0, df.columns[0], participant_ids)
    imputed_values.to_csv(imputed_values_output, index=False)

# usage
if __name__ == "__main__":
    input_csv = "HarvOx_subcortical_GM_volumes.csv"   # Replace with your input CSV file path
    output_csv = "subcortical_imputed.csv"  # Replace with your desired output CSV file path
    main(input_csv, output_csv, n_neighbors=5)
