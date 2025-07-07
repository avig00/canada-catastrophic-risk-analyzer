import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

df = pd.read_csv(os.path.join(OUTPUTS_DIR, "fsa_ml_dataset_labeled.csv"))
print(df['risk_label'].value_counts())

model = joblib.load(os.path.join(MODELS_DIR, "risk_model.pkl"))

X = df[['num_wildfires', 'total_area_burned', 'fires_per_100km2', 'burned_percent', 
    'num_tornadoes', 'wind_events_per_100km2']]
y = df['risk_label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))
print("Saved confusion matrix plot")
