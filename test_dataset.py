
import pandas as pd

# Load the downloaded dataset
data = pd.read_csv("dataset/mohler_asag.csv")

print("Dataset loaded successfully!")

print("Dataset shape:", data.shape)

print("\nDataset columns:")
print(data.columns.tolist())

print("\nFirst five records:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())
