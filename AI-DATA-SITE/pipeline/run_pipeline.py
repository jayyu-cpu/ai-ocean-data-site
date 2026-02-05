import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Allow running as: python -m pipeline.run_pipeline
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ------------------- Imports -------------------
from pipeline.fetch_noaa import fetch_noaa_crw, fetch_noaa_ph
from pipeline.fetch_allen import fetch_allen_coral_atlas
from pipeline.clean_transform import clean_noaa, clean_allen
from pipeline.merge_data import spatial_merge, integrate_ph

from ml.model import (
    health_score,
    detect_anomaly,
    train_lstm,
    forecast_lstm,
)

import backend.database as db
from backend.models import OceanMetrics
from sqlalchemy import insert

# ------------------- Config -------------------
FAST_MODE = True
MAX_ROWS_FAST = 5000

# ------------------- Pipeline -------------------
def run_daily_pipeline():
    """
    Daily Ocean Health Pipeline
    1. Fetch NOAA CRW (SST, DHW) + pH
    2. Fetch Allen Coral Atlas
    3. Clean & Transform
    4. Spatial Merge (PostGIS)
    5. ML Predictions
    6. Store to PostgreSQL (Core insert)
    """

    # Step 1: Fetch NOAA data
    print("Step 1: Fetching NOAA CRW data...")
    noaa = fetch_noaa_crw()
    ph = fetch_noaa_ph()

    # Step 2: Fetch Allen Coral Atlas
    print("Step 2: Fetching Allen Coral Atlas...")
    allen = fetch_allen_coral_atlas(noaa_df=noaa)

    # Step 3: Clean data
    print("Step 3: Cleaning data...")
    noaa = clean_noaa(noaa)
    allen = clean_allen(allen)

    # Step 4: Integrate pH
    print("Step 4: Integrating pH data...")
    merged = integrate_ph(noaa, ph)

    # Step 5: Spatial merge
    print("Step 5: Spatial merge with coral reefs...")
    merged = spatial_merge(merged, allen_gdf=allen)

    # Optional fast mode
    if FAST_MODE and len(merged) > MAX_ROWS_FAST:
        merged = merged.sample(MAX_ROWS_FAST, random_state=42)

    # Step 6: ML predictions
    print("Step 6: Running ML predictions...")
    merged["health_score"] = merged.apply(health_score, axis=1)
    merged["anomaly"] = detect_anomaly(merged["sst"])

    # Optional LSTM forecasting
    merged["forecast_ph"] = None
    if not FAST_MODE and merged["ph"].notna().sum() >= 30:
        try:
            model = train_lstm(merged["ph"].values)
            forecast = forecast_lstm(
                model,
                merged["ph"].values,
                steps_ahead=7
            )
            merged.loc[merged.index[-1], "forecast_ph"] = float(forecast[0])
        except Exception as e:
            print(f"LSTM forecast skipped: {e}")

    # Step 7: Store to PostgreSQL (FAST, CORE INSERT)
    print("Step 7: Storing to PostgreSQL...")
    db.init_db()

    records = merged[
        [
            "date",
            "lat",
            "lon",
            "sst",
            "dhw",
            "ph",
            "health_score",
            "anomaly",
            "forecast_ph",
        ]
    ].rename(
        columns={
            "lat": "latitude",
            "lon": "longitude",
        }
    ).to_dict(orient="records")

    print(f"Inserting {len(records)} rows into PostgreSQL...")

    with db.engine.begin() as conn:
        conn.execute(
            insert(OceanMetrics),
            records
        )

    print("PostgreSQL insert completed successfully")
    print("Pipeline completed successfully!")

# ------------------- Entry -------------------
if __name__ == "__main__":
    run_daily_pipeline()
