import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from config import OUTPUTS_DIR

# === Load dataset ===
df = pd.read_csv(f"{OUTPUTS_DIR}/fsa_ml_dataset.csv")

# === Feature set for clustering ===
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

# === KMeans clustering ===
kmeans = KMeans(n_clusters=3, random_state=42)
df['risk_cluster'] = kmeans.fit_predict(X_scaled)

# === Print cluster center sums ===
print("\nCluster center sums:")
for i, center in enumerate(kmeans.cluster_centers_):
    print(f"  Cluster {i}: {center.sum():.2f}")

# === Manual mapping based on printed values ===
risk_map = {
    0: 'Low',
    1: 'High',
    2: 'Medium'
}
df['risk_label'] = df['risk_cluster'].map(risk_map)

# === Save result ===
df.to_csv(f"{OUTPUTS_DIR}/fsa_ml_dataset_labeled.csv", index=False)
print("Labeled dataset saved to outputs/fsa_ml_dataset_labeled.csv")
