from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from app.model import predict_claim

app = FastAPI(title="Insurance Claim Prediction API")

# Define input schema
class ClaimInput(BaseModel):
    age: int
    vehicle_age: float
    vehicle_type: str
    accident_count: int
    weather_risk_index: float
    # Add more fields if needed to match trained model

@app.post("/predict")
def predict(data: ClaimInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    # Predict using model
    prediction = predict_claim(input_df, model_type="xgb")
    return {"predicted_claim_amount": round(prediction, 2)}
