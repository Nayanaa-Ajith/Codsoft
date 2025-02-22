import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset with explicit encoding and tab separator
try:
    df = pd.read_csv("IRIS.csv", encoding="utf-8", sep="\t")  # Use tab as separator
    print("Dataset loaded successfully!")
except UnicodeDecodeError:
    print("Error: UTF-8 encoding failed. Trying ISO-8859-1...")
    try:
        df = pd.read_csv("IRIS.csv", encoding="ISO-8859-1", sep="\t")
        print("Dataset loaded successfully with ISO-8859-1 encoding!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

# Ensure correct column names
print("Columns in dataset:", df.columns)

# Drop missing values
df = df.dropna()

# Extract features & target
try:
    X = df.iloc[:, :-1]  # Features (first 4 columns)
    y = df.iloc[:, -1]   # Target (last column)
except Exception as e:
    print(f"Error extracting features: {e}")
    exit()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
