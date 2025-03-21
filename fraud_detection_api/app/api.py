from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Load the trained model
model = joblib.load("../models/best_fraud_model.pkl")

# Initialize FastAPI
app = FastAPI()

# Define the input data model
class TransactionData(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    scaled_amount: float
    hour: float

@app.post("/predict")
async def predict_fraud(transaction: TransactionData):
    try:
        df = pd.DataFrame([transaction.dict()])
        expected_features = model.feature_names_in_
        df = df[expected_features]

        prediction = model.predict(df)[0]
        return {"fraud_prediction": "Fraud" if prediction == 1 else "Non-Fraud"}
    except Exception as e:
        return {"error": str(e)}
