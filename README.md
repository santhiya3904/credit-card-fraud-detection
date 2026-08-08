# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using a Random Forest Classifier.

## Project Overview

This project uses supervised machine learning to classify credit card transactions as either legitimate or fraudulent.

The trained Random Forest model is saved using Joblib and is used by the prediction script to classify a transaction.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Jupyter Notebook

## Machine Learning Model

- Random Forest Classifier
- Binary classification
- Output: FRAUD or LEGITIMATE

## Project Structure

```text
credit-card-fraud-detection/
├── models/
│   └── random_forest_fraud_model.pkl
├── src/
│   ├── fraud_detection.py
│   └── predict_fraud.py
├── .gitignore
├── README.md
└── requirements.txt

## Features

- Credit card fraud classification
- Random Forest machine learning model
- Saved trained model using Joblib
- Transaction prediction using Python

## How to Run

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
2. Make sure the dataset is available locally at:
    data/creditcard.csv
3. Run the prediction Script:
    python src/predict_fraud.py
The program will display whether the transaction is classified as:
FRAUD
LEGITIMATE
## Example Output

```text
Prediction: LEGITIMATE
Important Note
The credit card dataset is not included in this repository.
The dataset is excluded using .gitignore to avoid uploading the dataset to GitHub.

## Author

Santhiya
