import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("creditcard.csv")

# ✅ Scale the 'Amount' column
scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1,1))
df.drop(columns=["Amount"], inplace=True)  # ✅ Remove original 'Amount'

# ✅ Convert 'Time' into 'hour' and remove 'Time'
df["hour"] = (df["Time"] // 3600) % 24
df.drop(columns=["Time"], inplace=True)  # ✅ Remove original 'Time'

# ✅ Define Features & Target
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target variable

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Apply SMOTE for class balancing
smote = SMOTE(sampling_strategy=0.7, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ✅ Scale all features except 'scaled_amount' (already scaled) and 'hour' (categorical)
features_to_scale = [col for col in X.columns if col not in ["scaled_amount", "hour"]]
scaler = StandardScaler()
X_train_smote[features_to_scale] = scaler.fit_transform(X_train_smote[features_to_scale])
X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])

# ✅ Train XGBoost Model
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
xgb.fit(X_train_smote, y_train_smote)

# ✅ Save the trained model
joblib.dump(xgb, "fraud_detection_xgb.pkl")  # ✅ Make sure you save `xgb`, not `best_xgb`
print("✅ Model saved successfully!")

# ✅ Make predictions
y_pred_xgb = xgb.predict(X_test)

# ✅ Evaluate performance
print("\nXGBoost Model Performance:\n")
print(classification_report(y_test, y_pred_xgb))
