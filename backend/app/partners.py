from pathlib import Path
import pandas as pd
from datetime import datetime

PARTNERS_CSV = Path("backend/data/partners.csv")
DONATIONS_LOG = Path("backend/data/donations_log.csv")

def load_partners() -> pd.DataFrame:
    if not PARTNERS_CSV.exists():
        raise FileNotFoundError(f"Partners CSV not found at {PARTNERS_CSV}")
    df = pd.read_csv(PARTNERS_CSV)
    # normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def partners_by_region(region: str | None = None) -> list[dict]:
    df = load_partners()
    if region:
        df = df[df["region"].str.lower() == region.lower()]
    return df.to_dict(orient="records")

def append_donation_record(rec: dict) -> dict:
    DONATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "timestamp": datetime.utcnow().isoformat(),
        **rec
    }])
    if DONATIONS_LOG.exists():
        df_existing = pd.read_csv(DONATIONS_LOG)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(DONATIONS_LOG, index=False)
    return rec
