import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from config import OUTPUTS_DIR, MODELS_DIR

# === Load data and model ===
df = pd.read_csv(f"{OUTPUTS_DIR}/fsa_ml_dataset_labeled.csv")
model = joblib.load(f"{MODELS_DIR}/risk_model.pkl")

# === Define features used in training ===
features = [
    'num_wildfires',
    'total_area_burned',
    'fires_per_100km2',
    'burned_percent',
    'num_tornadoes',
    'wind_events_per_100km2'
]
X = df[features]

# === Create SHAP explainer and compute values ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X, check_additivity=False)

# === SHAP Bar Plot (default color scheme) ===
plt.figure()
shap.summary_plot(shap_values, X, show=False, plot_type="bar")
plt.savefig(f"{OUTPUTS_DIR}/global_shap_summary_bar.png", bbox_inches="tight")
print("Saved SHAP bar plot")

# === SHAP Beeswarm Plot (default color scheme) ===
plt.figure()
shap.summary_plot(shap_values, X, show=False, plot_type="dot")
plt.savefig(f"{OUTPUTS_DIR}/global_shap_summary_beeswarm.png", bbox_inches="tight")
print("Saved SHAP beeswarm plot")
