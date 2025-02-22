import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Initialize dropdown values
sepal_length_values = []
sepal_width_values = []
petal_length_values = []
petal_width_values = []

# Load dataset with error handling
try:
    df = pd.read_csv("IRIS.csv", encoding="utf-8", delimiter="\t")  # FIXED: Set tab delimiter
    
    print("🔹 Dataset Loaded Successfully")
    print(df.head())  # Show first few rows for verification
    print("\n🔹 Columns Found:", df.columns.tolist())

    # Ensure correct column names
    expected_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"❌ Dataset column mismatch! Found: {list(df.columns)}")

    # Load unique values for dropdown lists
    sepal_length_values = sorted(df["sepal_length"].dropna().unique().tolist())
    sepal_width_values = sorted(df["sepal_width"].dropna().unique().tolist())
    petal_length_values = sorted(df["petal_length"].dropna().unique().tolist())
    petal_width_values = sorted(df["petal_width"].dropna().unique().tolist())

    print("✅ Dropdown values loaded!")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# Load trained model
try:
    with open("iris_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Home route - Display dropdowns
@app.route("/")
def index():
    return render_template(
        "index.html",
        sepal_length_values=sepal_length_values,
        sepal_width_values=sepal_width_values,
        petal_length_values=petal_length_values,
        petal_width_values=petal_width_values,
        prediction=None  # Initially, no prediction
    )

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return "❌ Model not loaded. Please check the model file."

        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Make prediction
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

        return render_template(
            "index.html",
            prediction=prediction,
            sepal_length_values=sepal_length_values,
            sepal_width_values=sepal_width_values,
            petal_length_values=petal_length_values,
            petal_width_values=petal_width_values,
        )
    
    except Exception as e:
        return f"❌ Error in prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
