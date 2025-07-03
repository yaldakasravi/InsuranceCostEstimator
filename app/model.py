import joblib
import pandas as pd
import os

# Load trained models
xgb_model_path = os.path.join("models", "xgb_model.pkl")
lgbm_model_path = os.path.join("models", "lgbm_model.pkl")

xgb_model = joblib.load(xgb_model_path)
lgbm_model = joblib.load(lgbm_model_path)

# Inference function
def predict_claim(input_data: pd.DataFrame, model_type: str = "xgb") -> float:
    model = xgb_model if model_type == "xgb" else lgbm_model
    prediction = model.predict(input_data)
    return float(prediction[0])
