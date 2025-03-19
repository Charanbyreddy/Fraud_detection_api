import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# ✅ Load dataset
df = pd.read_csv("creditcard.csv")

# ✅ Scale 'Amount' column
scaler = StandardScaler()
df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df.drop(columns=["Amount"], inplace=True)  

# ✅ Convert 'Time' to 'hour' and remove 'Time'
df["hour"] = (df["Time"] // 3600) % 24
df.drop(columns=["Time"], inplace=True)  

# ✅ Define Features & Target
X = df.drop(columns=["Class"])  
y = df["Class"]  

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Apply SMOTE for class balancing
smote = SMOTE(sampling_strategy=0.7, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ✅ Scale all features except 'scaled_amount' and 'hour'
features_to_scale = [col for col in X.columns if col not in ["scaled_amount", "hour"]]
scaler = StandardScaler()
X_train_smote[features_to_scale] = scaler.fit_transform(X_train_smote[features_to_scale])
X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])

# ✅ Initialize Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=100, verbose=0, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

# ✅ Train and Evaluate Models
best_model = None
best_f1_score = 0

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    report = classification_report(y_test, y_pred, output_dict=True)
    f1 = report["1"]["f1-score"]  # F1-score for the fraud class
    print(classification_report(y_test, y_pred))
    
    # Save the best model based on F1-score
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model

# ✅ Save the best model
joblib.dump(best_model, "best_fraud_model.pkl")
print("\n✅ Best model saved successfully!")
 
