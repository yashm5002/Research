# Create: C:\SEED-VII_Project\utils\data_diagnostic.py
import numpy as np
import pandas as pd

# Load the data
X_train = np.load('C:/SEED-VII_Project/data/final/X_train.npy')
y_train = np.load('C:/SEED-VII_Project/data/final/y_discrete_train.npy')
X_test = np.load('C:/SEED-VII_Project/data/final/X_test.npy')
y_test = np.load('C:/SEED-VII_Project/data/final/y_discrete_test.npy')

print("=== DATA INTEGRITY CHECK ===")
print(f"Training features shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Check feature statistics
print(f"\nFeature statistics:")
print(f"Training features - Mean: {X_train.mean():.6f}, Std: {X_train.std():.6f}")
print(f"Training features - Min: {X_train.min():.6f}, Max: {X_train.max():.6f}")
print(f"Any NaN values: {np.isnan(X_train).any()}")
print(f"Any infinite values: {np.isinf(X_train).any()}")

# Check label distribution
print(f"\nLabel distribution:")
unique_labels, counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples ({count/len(y_train)*100:.1f}%)")

# Check if labels are properly encoded (0-6)
print(f"\nLabel encoding check:")
print(f"Min label: {y_train.min()}")
print(f"Max label: {y_train.max()}")
print(f"Unique labels: {sorted(unique_labels)}")
