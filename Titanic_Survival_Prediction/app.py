from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
with open('titanic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        Pclass = int(request.form['Pclass'])
        Age = request.form['Age']  # Could be empty or non-numeric
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = request.form['Fare']  # Could be empty or non-numeric
        Sex = request.form['Sex']  # 'male' or 'female'
        Embarked = request.form['Embarked']  # 'C', 'Q', or 'S'

        # Convert Age and Fare to numeric values, handling empty or invalid input
        Age = pd.to_numeric(Age, errors='coerce')  # Converts invalid values to NaN
        Fare = pd.to_numeric(Fare, errors='coerce')

        # Replace missing or invalid values with appropriate defaults
        if pd.isnull(Age):
            Age = 30  # Default average age
        if pd.isnull(Fare):
            Fare = 35  # Default average fare

        # Encode the 'Sex' feature
        Sex_male = 1 if Sex == 'male' else 0

        # Encode 'Embarked' with two variables only (drop one to avoid the dummy variable trap)
        Embarked_C = 1 if Embarked == 'C' else 0
        Embarked_Q = 1 if Embarked == 'Q' else 0
        # We drop Embarked_S because it is redundant (if neither C nor Q, it's S implicitly)

        # Prepare input features (exactly 7 features to match model training)
        input_features = np.array([
            Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_C  # **Only 7 features**
        ]).reshape(1, -1)

        # Scale the features
        input_features_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_features_scaled)

        # Interpret result
        result = 'Survived' if prediction[0] == 1 else 'Not Survived'
        return render_template('index.html', prediction_text=f'The person is predicted to: {result}')
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
