from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
import os

app = FastAPI()

# URL where the model is stored (Replace with your actual Google Drive/Hugging Face URL)
MODEL_URL = "https://your-storage-link.com/best_fraud_model.pkl"
MODEL_PATH = "models/best_fraud_model.pkl"

# Ensure the model is downloaded
if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)  # Create models folder if not exists
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# Load the trained model
model = joblib.load(MODEL_PATH)


# Define input data format using Pydantic
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

        # Ensure feature order matches model training data
        expected_features = model.feature_names_in_
        df = df[expected_features]

        # Make a prediction
        prediction = model.predict(df)[0]

        return {"fraud_prediction": "Fraud" if prediction == 1 else "Non-Fraud"}
    
    except Exception as e:
        return {"error": str(e)}
