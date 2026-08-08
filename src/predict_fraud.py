import joblib
import pandas as pd

# Load the trained Random Forest model
model = joblib.load("models/random_forest_fraud_model.pkl")

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Select one transaction for testing
transaction = df.drop("Class", axis=1).iloc[[0]]

# Make prediction
prediction = model.predict(transaction)[0]

# Display result
if prediction == 1:
    print("Prediction: FRAUD ⚠️")
else:
    print("Prediction: LEGITIMATE ✅")