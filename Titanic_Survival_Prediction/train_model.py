# train_model.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# Load your Titanic dataset (replace this with the correct path)
data = pd.read_csv('titanic_data.csv')

# Select features and target (adjust this according to your dataset's columns)
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']]
y = data['Survived']

# Define the preprocessing steps: scaling for numeric columns and one-hot encoding for categorical ones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('cat', OneHotEncoder(), ['Sex', 'Embarked'])
    ])

# Create a pipeline with preprocessing and logistic regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit the model on the training data
model.fit(X, y)

# Save the trained model
with open('titanic_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# If you also want to use the scaler separately, you can save it similarly
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(preprocessor, scaler_file)
