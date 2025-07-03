import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

# Load your dataset
data = pd.read_csv("data/sample_data.csv")

# Define features and target
features = data.drop(columns=["claim_amount"])
target = data["claim_amount"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# Train LightGBM model
lgbm_model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
lgbm_model.fit(X_train, y_train)

# Evaluate
for name, model in [("XGBoost", xgb_model), ("LightGBM", lgbm_model)]:
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(lgbm_model, "models/lgbm_model.pkl")
