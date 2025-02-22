import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('D:/Titanic-Dataset.csv')

# Fill missing values for 'Age' and 'Embarked'
data['Age'] = data['Age'].fillna(data['Age'].mean())  # Fill missing Age with mean value
data['Embarked'].fillna(data['Embarked'].mode()[0])  # Fill missing Embarked with mode value

# One-hot encode categorical columns: 'Sex' and 'Embarked'
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Check the columns in the dataset
print("Dataset Columns:", data.columns)

# Define the features (X) and target (y)
# Now using 'Embarked_S' and 'Embarked_Q' instead of 'Embarked_C'
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
