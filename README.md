# Credit Card Fraud Detection

## Project Overview

This project is a Machine Learning based Credit Card Fraud Detection system. It classifies credit card transactions as either **FRAUD** or **LEGITIMATE** using a Random Forest classification model.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## Machine Learning Model

A **Random Forest Classifier** is used to detect fraudulent credit card transactions.

The trained model is saved using **Joblib** and used for predicting new transactions.

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
```

## Features

* Credit card fraud classification
* Random Forest machine learning model
* Saved trained model using Joblib
* Transaction prediction using Python

## How to Run

### 1. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 2. Make sure the dataset is available locally

```text
data/creditcard.csv
```

### 3. Run the prediction script

```bash
python src/predict_fraud.py
```

The program will display whether the transaction is classified as:

```text
FRAUD
```

or

```text
LEGITIMATE
```

## Example Output

```text
Prediction: LEGITIMATE
```

## Important Note

The credit card dataset is not included in this repository.

The dataset is excluded using `.gitignore` to avoid uploading the dataset to GitHub.

## Author

Santhiya
