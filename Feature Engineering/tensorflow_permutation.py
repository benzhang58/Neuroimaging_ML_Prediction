import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess data
def load_data(file_path, target_column):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Handle empty cells by filling them with the mean of the column
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)
    
    return X, y

# Create a simple neural network model
def create_model(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # For regression
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to calculate permutation importance
def permutation_importance(model, X_val, y_val, metric_func):
    baseline_score = metric_func(y_val, model.predict(X_val))
    importances = []
    
    for i in range(X_val.shape[1]):
        X_permuted = X_val.copy()
        np.random.shuffle(X_permuted[:, i])
        score = metric_func(y_val, model.predict(X_permuted))
        importances.append(baseline_score - score)
        
    return np.array(importances)

# Load data
X, y = load_data('All_features.csv', 'stand_IQ')

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train the model
model = create_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# Evaluate permutation importance
importances = permutation_importance(model, X_val, y_val, r2_score)
feature_importances = pd.Series(importances, index=X.columns)
print("Feature Importances:\n", feature_importances.sort_values(ascending=False))
