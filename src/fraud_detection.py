import pandas as pd

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Display basic information
print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

# Display first 5 transactions
print("\nFirst 5 transactions:")
print(df.head())
# Check transaction classes
print("\nTransaction class distribution:")
print(df["Class"].value_counts())

print("\nFraud percentage:")
print(df["Class"].value_counts(normalize=True) * 100)
# Check for missing values
print("\nMissing values:")
print(df.isnull().sum().sum())
# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)
from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Create Logistic Regression pipeline
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])

# Train the model
print("\nTraining Logistic Regression model...")
logistic_model.fit(X_train, y_train)

print("Logistic Regression model trained successfully!")
from sklearn.metrics import classification_report, confusion_matrix

# Make predictions on test data
y_pred = logistic_model.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train the model
print("\nTraining Random Forest model...")
random_forest_model.fit(X_train, y_train)

print("Random Forest model trained successfully!")

# Make predictions
rf_pred = random_forest_model.predict(X_test)

# Evaluate Random Forest
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
from sklearn.metrics import f1_score, recall_score, precision_score

print("\nModel Comparison:")
print("--------------------------------")

print("Logistic Regression")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

print("\nRandom Forest")
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1-Score:", f1_score(y_test, rf_pred))
import joblib

# Save the trained Random Forest model
joblib.dump(random_forest_model, "models/random_forest_fraud_model.pkl")

print("\nRandom Forest model saved successfully!")