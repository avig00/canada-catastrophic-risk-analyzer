import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sqlalchemy import create_engine

# === Load FSA Shapefile ===
print("Loading FSA shapefile...")
fsa = gpd.read_file("data/fsa/lfsa000b21a_e.shp").to_crs(epsg=4326)

# === Load Wind Event CSV ===
print("Loading wind event data...")
wind_df = pd.read_csv("data/wind/NTP_Event_Summaries_stakeholder_-1669238883341502514.csv")

# === Filter and Convert to GeoDataFrame ===
wind_df = wind_df.dropna(subset=["x", "y"])  # remove missing coords
wind_gdf = gpd.GeoDataFrame(
    wind_df,
    geometry=gpd.points_from_xy(wind_df["x"], wind_df["y"]),
    crs="EPSG:4326"
)

# === Spatial Join to FSA ===
print("Performing spatial join (wind events to FSAs)...")
joined = gpd.sjoin(wind_gdf, fsa[['CFSAUID', 'PRUID', 'PRNAME', 'LANDAREA', 'geometry']], how='inner', predicate='within')

# === Aggregate Wind Features per FSA ===
print("Aggregating wind features...")
summary = joined.groupby("CFSAUID").agg(
    PRUID=('PRUID', 'first'),
    PRNAME=('PRNAME', 'first'),
    LANDAREA=('LANDAREA', 'first'),
    num_wind_events=('Event Name', 'count'),
    num_tornadoes=('Event Type', lambda x: (x == 'tornado_over_land').sum()),
    avg_max_wind=('Maximum Wind Speed (km/h)', 'mean'),
    max_max_wind=('Maximum Wind Speed (km/h)', 'max'),
    avg_path_length_km=('Track Length (km)', 'mean'),
    avg_path_width_m=('Maximum Path Width (m)', 'mean')
).reset_index()
summary["wind_events_per_100km2"] = summary["num_wind_events"] / (summary["LANDAREA"] / 100)

# === Save to SQLite DB ===
engine = create_engine("sqlite:///data/fsa_risk.db")  # same DB as wildfire
summary.to_sql("wind_features", engine, index=False, if_exists="replace")
print("Saved to SQLite: data/fsa_risk.db (table: wind_features)")
