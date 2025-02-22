import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset with correct encoding
df = pd.read_csv("data/movies.csv", encoding="ISO-8859-1")

# Print column names for debugging
print("Columns in dataset:", df.columns)

# Standardize column names (strip spaces and make lowercase)
df.columns = df.columns.str.strip().str.lower()

# Rename columns to match dataset structure
column_mapping = {
    "genre": "genre",
    "director": "director",
    "actor 1": "actor_1",
    "actor 2": "actor_2",
    "actor 3": "actor_3",
    "rating": "imdb_rating"  # Adjusted based on your dataset
}

df = df.rename(columns=column_mapping)

# Combine actors into one column (optional)
df["actors"] = df["actor_1"] + ", " + df["actor_2"] + ", " + df["actor_3"]

# Select relevant features and drop missing values
df = df[['genre', 'director', 'actors', 'imdb_rating']].dropna()

# Encode categorical variables
label_encoders = {}
for col in ['genre', 'director', 'actors']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save encoders for later use
with open("models/encoder.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Split dataset
X = df[['genre', 'director', 'actors']]
y = df['imdb_rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Save model
with open("models/movie_rating_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete. Saved as 'models/movie_rating_model.pkl'.")
