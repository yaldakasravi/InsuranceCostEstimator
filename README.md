# Insurance Claim Amount Predictor

This project builds an end-to-end machine learning system to predict insurance claim amounts based on historical policyholder data.

## Features

- Trains two models (XGBoost and LightGBM) on real-world structured data
- Evaluates model accuracy with RMSE and MAE
- Exposes prediction endpoint via FastAPI
- Uses SHAP to explain model behavior and feature importance
- Ready for cloud deployment (e.g., AWS EC2)

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
Train models:
python train_model.py
Run API server:
uvicorn app.main:app --reload
Send a POST request to /predict with JSON payload:
{
  "age": 45,
  "vehicle_age": 5.0,
  "vehicle_type": "SUV",
  "accident_count": 2,
  "weather_risk_index": 0.6
}
SHAP Analysis

Run SHAP visualization script:

python shap_analysis.py
