import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function to load data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to compute the correlation matrix and plot the heatmap
def plot_correlation_heatmap(data, figsize=(20, 16), annot=False, cmap='coolwarm', annot_fontsize=8):
    # Compute the correlation matrix
    corr_matrix = data.corr()

    # Plot the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=".2f", square=True, 
                cbar_kws={'shrink': .8}, annot_kws={"size": annot_fontsize})
    
    # Adjust the label font size
    plt.xticks(fontsize=5)  # Shrink the x-axis labels
    plt.yticks(fontsize=5)  # Shrink the y-axis labels
    
    plt.title('Correlation Heatmap', fontsize=12)  # Adjust the title font size
    plt.show()
    
def main():
    file_path = 'HarvOx_Right_cortical_GM_volumes.csv'  # Replace with your actual CSV file path
    data = load_data(file_path)

    # Optionally, you might want to drop non-numeric columns before calculating correlations
    data = data.select_dtypes(include=[np.number])

    plot_correlation_heatmap(data, annot=True, annot_fontsize=3)  # Adjust annot_fontsize to control the font size of correlation values

if __name__ == "__main__":
    main()
