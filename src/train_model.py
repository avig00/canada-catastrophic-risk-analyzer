import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from config import OUTPUTS_DIR, MODELS_DIR

# === Load data ===
df = pd.read_csv(f"{OUTPUTS_DIR}/fsa_ml_dataset_labeled.csv")

# === Select features and labels ===
features = ['num_wildfires', 'total_area_burned', 'fires_per_100km2', 'burned_percent', 
           'num_tornadoes', 'wind_events_per_100km2']
X = df[features]
y = df['risk_label']

# === Train/test split with stratification ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === Apply SMOTE only on training data ===
smote = SMOTE(k_neighbors=2, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# === Train model with class weighting ===
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)

# === Save model ===
joblib.dump(model, f"{MODELS_DIR}/risk_model.pkl")
print("Model saved to models/risk_model.pkl")
