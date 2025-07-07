import sqlite3
import pandas as pd
from config import DB_PATH, OUTPUTS_DIR

conn = sqlite3.connect(DB_PATH)
wildfire = pd.read_sql("SELECT * FROM wildfire_features", conn)
wind = pd.read_sql("SELECT * FROM wind_features", conn)

df = pd.merge(wildfire, wind, on="CFSAUID", suffixes=("_fire", "_wind"))
df.to_csv(f"{OUTPUTS_DIR}/fsa_ml_dataset.csv", index=False)
print("Saved outputs/fsa_ml_dataset.csv")
