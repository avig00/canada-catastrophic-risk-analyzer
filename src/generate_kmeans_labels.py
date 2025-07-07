import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from config import OUTPUTS_DIR

# === Load dataset ===
df = pd.read_csv(f"{OUTPUTS_DIR}/fsa_ml_dataset.csv")

# === Finalized feature set for clustering ===
features = [
    'num_wildfires',
    'total_area_burned',
    'fires_per_100km2',
    'burned_percent',
    'num_tornadoes',
    'wind_events_per_100km2'
]

# === Drop rows with missing values in clustering features ===
df = df.dropna(subset=features)

# === Scale features ===
X_scaled = StandardScaler().fit_transform(df[features])

# === Apply KMeans clustering ===
kmeans = KMeans(n_clusters=3, random_state=42)
df['risk_cluster'] = kmeans.fit_predict(X_scaled)

# === Map clusters to ordered risk labels (Low < Medium < High) ===
cluster_order = kmeans.cluster_centers_.sum(axis=1).argsort()
risk_map = {cluster_order[0]: 'Low', cluster_order[1]: 'Medium', cluster_order[2]: 'High'}
df['risk_label'] = df['risk_cluster'].map(risk_map)

# === Save labeled dataset ===
df.to_csv(f"{OUTPUTS_DIR}/fsa_ml_dataset_labeled.csv", index=False)
print("Saved labeled dataset with risk clusters and risk_label to outputs/fsa_ml_dataset_labeled.csv")
