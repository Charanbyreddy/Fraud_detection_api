from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ✅ Load the updated best model
model = joblib.load("best_fraud_model.pkl")  # 🔥 Update this line to use the best model

# ✅ Create FastAPI instance
app = FastAPI()

# ✅ Define input data format using Pydantic
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

# ✅ Define API endpoint
@app.post("/predict")
async def predict_fraud(transaction: TransactionData):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([transaction.dict()])

        # Make prediction
        prediction = model.predict(df)[0]

        # Return result
        return {"fraud_prediction": "Fraud" if prediction == 1 else "Non-Fraud"}
    
    except Exception as e:
        return {"error": str(e)}
