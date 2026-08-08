Project Overview
This project implements a supervised machine learning classification approach to detect fraudulent credit card transactions.
A Random Forest Ensemble Classifier is trained to perform binary classification on transaction data, categorizing each transaction as either legitimate (non-fraudulent) or fraudulent. The trained model is serialized (persisted) and used by a standalone prediction script to perform real-time inference on individual transaction records.
Tech Stack
Python – core programming language
Pandas – data manipulation & preprocessing
NumPy – numerical computation
Scikit-learn – model implementation & evaluation metrics
Random Forest (Ensemble Learning algorithm) – classification model
Joblib – model serialization/deserialization (persistence)
Jupyter Notebook – exploratory data analysis (EDA) & experimentation environment
