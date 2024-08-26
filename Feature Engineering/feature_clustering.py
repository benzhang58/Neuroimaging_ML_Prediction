import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Load your data
def load_data(file_path):
    return pd.read_csv(file_path)

# Calculate the correlation of each feature with the target
def calculate_top_correlated_features(df, target_column, top_n=30):
    correlations = df.corr()[target_column].drop(target_column).abs()  # Get absolute correlations with the target
    top_features = correlations.nlargest(top_n).index.tolist()  # Get the top N features
    return top_features, correlations[top_features]

# Cluster features based on correlation and select a representative from each cluster
def cluster_and_select_features(df, top_features, target_column, threshold=0.5):
    corr_matrix = df[top_features].corr().abs()  # Absolute value of correlation matrix
    
    # Convert the correlation matrix to a distance matrix
    distance_matrix = 1 - corr_matrix
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(squareform(distance_matrix), method='average')
    
    # Form clusters based on the threshold
    cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
    
    # Select one feature from each cluster
    selected_features = []
    for cluster in np.unique(cluster_labels):
        cluster_features = [top_features[i] for i in range(len(top_features)) if cluster_labels[i] == cluster]
        # Select the feature with the highest correlation with the target
        best_feature = max(cluster_features, key=lambda x: df[x].corr(df[target_column]))
        selected_features.append(best_feature)
    
    return selected_features, cluster_labels

def main():
    file_path = 'All_features.csv'  # Replace with your actual file path
    target_column = 'stand_IQ'  # Replace with your actual target column name
    
    df = load_data(file_path)
    
    # Step 1: Calculate top 30 correlated features with the target
    top_features, correlations = calculate_top_correlated_features(df, target_column)
    print("Top 30 features correlated with the target:")
    print(correlations)
    
    # Step 2: Cluster correlated features and select a representative from each cluster
    selected_features, cluster_labels = cluster_and_select_features(df, top_features, target_column)
    print("\nSelected features from each cluster:")
    print(selected_features)
    
    print("\nFeature clusters:")
    for cluster in np.unique(cluster_labels):
        cluster_features = [top_features[i] for i in range(len(top_features)) if cluster_labels[i] == cluster]
        print(f"Cluster {cluster}: {cluster_features}")

if __name__ == "__main__":
    main()
