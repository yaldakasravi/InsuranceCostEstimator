import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("models/xgb_model.pkl")
data = pd.read_csv("data/sample_data.csv")
features = data.drop(columns=["claim_amount"])

# Create SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(features)

# SHAP summary plot
shap.plots.beeswarm(shap_values, max_display=15)
plt.tight_layout()
plt.show()
