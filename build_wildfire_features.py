import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine

# === Load Data ===
print("Loading shapefiles...")
fsa = gpd.read_file("data/fsa/lfsa000b21a_e.shp").to_crs(epsg=4326)
wildfire = gpd.read_file("data/wildfire/NFDB_poly_20210707.shp").to_crs(epsg=4326)

# === Optional: Filter wildfires (e.g., last 20 years only) ===
wildfire = wildfire[wildfire['YEAR'] >= 2000]
print(f"Filtered to {len(wildfire)} wildfire records since 2000.")

# === Keep required columns only ===
wildfire = wildfire[['geometry', 'YEAR', 'SIZE_HA']]

# === Spatial Join ===
print("Performing spatial join (wildfire polygons to FSAs)...")
joined = gpd.sjoin(wildfire, fsa[['CFSAUID', 'PRUID', 'PRNAME', 'LANDAREA', 'geometry']], how='inner', predicate='intersects')

# === Aggregate Wildfire Stats Per FSA ===
print("Aggregating wildfire features...")
summary = joined.groupby("CFSAUID").agg(
    PRUID=('PRUID', 'first'),
    PRNAME=('PRNAME', 'first'),
    LANDAREA=('LANDAREA', 'first'),
    num_wildfires=('YEAR', 'count'),
    total_area_burned=('SIZE_HA', 'sum'),
    avg_area_burned=('SIZE_HA', 'mean'),
    years_with_fires=('YEAR', lambda x: x.nunique())
).reset_index()

summary["fires_per_100km2"] = summary["num_wildfires"] / (summary["LANDAREA"] / 100)
summary["burned_percent"] = (summary["total_area_burned"] / summary["LANDAREA"]) * 100


# === Save to SQLite database ===
engine = create_engine("sqlite:///data/fsa_risk.db")  # creates database file
summary.to_sql("wildfire_features", engine, index=False, if_exists="replace")
print("Saved to SQLite: data/fsa_risk.db (table: wildfire_features)")
