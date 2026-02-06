import os
import requests
import xarray as xr
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

SST_BASE = os.getenv(
    "NOAA_SST_BASE_URL",
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/v3.1_op/nc/v1.0/daily/sst"
)
DHW_BASE = os.getenv(
    "NOAA_DHW_BASE_URL",
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/v3.1_op/nc/v1.0/daily/dhw"
)

def _candidate_dates(days_back=3):
    today = date.today()
    return [today - timedelta(days=i) for i in range(days_back)]

def _download(url, path):
    if os.path.exists(path):
        return True
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
        return False
    except Exception:
        return False

def _download_latest(base_url, fname_tmpl, prefix):
    for d in _candidate_dates():
        fname = fname_tmpl.format(date=d.strftime("%Y%m%d"))
        url = f"{base_url}/{d.year}/{fname}"
        path = f"{prefix}_{d.strftime('%Y%m%d')}.nc"
        if _download(url, path):
            return path, d
    return None, None

def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def fetch_noaa_crw():
    """
    Fetch NOAA CRW daily SST (CoralTemp) + DHW.
    """
    # SST (CoralTemp) file name format:
    # coraltemp_v3.1_YYYYMMDD.nc
    sst_path, sst_date = _download_latest(
        SST_BASE, "coraltemp_v3.1_{date}.nc", "NOAA_SST"
    )

    # DHW file name format:
    # ct5km_dhw_v3.1_YYYYMMDD.nc
    dhw_path, _ = _download_latest(
        DHW_BASE, "ct5km_dhw_v3.1_{date}.nc", "NOAA_DHW"
    )

    if not sst_path or not os.path.exists(sst_path):
        # fallback demo data
        return pd.DataFrame(
            {
                "lat": [6.5, 6.6, 6.7],
                "lon": [92.5, 92.6, 92.7],
                "sst": [28.2, 28.4, 28.3],
                "dhw": [0.5, 0.6, 0.7],
                "date": [date.today()] * 3,
            }
        )

    ds_sst = xr.open_dataset(sst_path, engine="h5netcdf")
    df_sst = ds_sst.to_dataframe().reset_index()

    # Find sst variable name safely
    sst_col = _find_col(df_sst, ["sst", "analysed_sst", "sea_surface_temperature"])
    if not sst_col:
        raise RuntimeError("Could not find SST variable in NOAA SST dataset")

    df_sst = df_sst[["lat", "lon", sst_col]].rename(columns={sst_col: "sst"})
    df_sst["date"] = sst_date or date.today()

    # DHW
    if dhw_path and os.path.exists(dhw_path):
        ds_dhw = xr.open_dataset(dhw_path,engine="h5netcdf")
        df_dhw = ds_dhw.to_dataframe().reset_index()
        dhw_col = _find_col(df_dhw, ["dhw", "degree_heating_week"])
        if dhw_col:
            df_dhw = df_dhw[["lat", "lon", dhw_col]].rename(columns={dhw_col: "dhw"})
            df_sst = df_sst.merge(df_dhw, on=["lat", "lon"], how="left")

    # Default if missing
    if "dhw" not in df_sst.columns:
        df_sst["dhw"] = 0.0

    return df_sst

def fetch_noaa_ph():
    """
    Fetch pH data (optional; still fallback to demo if missing).
    """
    ph_path = "NOAA_PH_FILE.nc"
    ph_url = os.getenv("NOAA_PH_URL", "")

    if ph_url and not os.path.exists(ph_path):
        try:
            r = requests.get(ph_url, timeout=30)
            r.raise_for_status()
            with open(ph_path, "wb") as f:
                f.write(r.content)
        except Exception:
            pass

    if not os.path.exists(ph_path):
        return pd.DataFrame(
            {
                "lat": [6.5, 6.6, 6.7],
                "lon": [92.5, 92.6, 92.7],
                "ph": [8.10, 8.11, 8.09],
                "date": [date.today()] * 3,
            }
        )

    ph_ds = xr.open_dataset(ph_path,engine="h5netcdf")
    ph_df = ph_ds.to_dataframe().reset_index()
    ph_df = ph_df[["lat", "lon", "ph"]]
    ph_df["date"] = date.today()
    return ph_df
