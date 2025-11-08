"""
Deadstock Dilemma - Core data loader and business logic.

- Robust path resolution using DATA_DIR (or backend/data)
- Inventory CSV: processed_inventory.csv (required)
- Partners CSV: partners.csv (optional but recommended)
- Donations log: donations_log.csv (auto-created if missing)
- Similarity: TF-IDF over a text_feature composed from item meta
- Redistribution: simple demand-gap heuristic with tunable tolerance
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Columns:
    product_id: str = "product_id"
    category: str = "category"
    region: str = "region"
    inventory: str = "inventory_level"
    sold: str = "units_sold"
    forecast: str = "demand_forecast"
    price: str = "price"
    discount: str = "discount"
    weather: str = "weather_condition"
    promo: str = "holiday/promotion"
    competitor: str = "competitor_pricing"
    text_feature: str = "text_feature"  # created at load if missing


class DeadstockRecommender:
    def __init__(self) -> None:
        # Base directory = backend/ (this file is backend/app/recommender.py)
        base_dir = Path(__file__).resolve().parent.parent
        # Prefer an external DATA_DIR (Render), else repo folder backend/data
        self.data_dir = Path(os.getenv("DATA_DIR", base_dir / "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.cols = Columns()

        self.inventory_csv = self.data_dir / "processed_inventory.csv"
        self.partners_csv = self.data_dir / "partners.csv"
        self.donations_log = self.data_dir / "donations_log.csv"

        if not self.inventory_csv.exists():
            raise FileNotFoundError(
                f"Inventory CSV missing: {self.inventory_csv}\n"
                "Ensure backend/data/processed_inventory.csv is committed, "
                "or DATA_DIR points to a folder that contains it."
            )

        # Load and normalize inventory
        self.df: pd.DataFrame = pd.read_csv(self.inventory_csv)
        self._normalize_columns()
        self._ensure_text_feature()
        self._build_tfidf()

        # Load partners (optional)
        if self.partners_csv.exists():
            self.partners_df = pd.read_csv(self.partners_csv)
        else:
            self.partners_df = pd.DataFrame(columns=["region", "ngo", "contact"])

        # Ensure donation log exists
        if not self.donations_log.exists():
            pd.DataFrame(
                columns=["timestamp", "product_id", "qty", "region", "partner_name", "contact", "notes"]
            ).to_csv(self.donations_log, index=False)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _normalize_columns(self) -> None:
        """
        Make columns snake_case and create fallbacks for known variants.
        """
        # Lower + replace spaces/strange chars
        self.df.columns = (
            self.df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.replace("/", "_")
        )

        # Map common variants into our canonical names
        rename_map = {}
        m = self.cols

        # Product id
        for cand in ["product_id", "productid", "pid"]:
            if cand in self.df.columns:
                rename_map[cand] = m.product_id
                break

        # Category
        for cand in ["category", "cat"]:
            if cand in self.df.columns:
                rename_map[cand] = m.category
                break

        # Region
        for cand in ["region", "store_region", "market"]:
            if cand in self.df.columns:
                rename_map[cand] = m.region
                break

        # Inventory
        for cand in ["inventory_level", "inventory", "stock_on_hand"]:
            if cand in self.df.columns:
                rename_map[cand] = m.inventory
                break

        # Units sold
        for cand in ["units_sold", "sold", "sales_units"]:
            if cand in self.df.columns:
                rename_map[cand] = m.sold
                break

        # Demand forecast
        for cand in ["demand_forecast", "forecast", "predicted_demand"]:
            if cand in self.df.columns:
                rename_map[cand] = m.forecast
                break

        # Price
        for cand in ["price", "unit_price"]:
            if cand in self.df.columns:
                rename_map[cand] = m.price
                break

        # Discount
        for cand in ["discount", "discount_pct", "markdown"]:
            if cand in self.df.columns:
                rename_map[cand] = m.discount
                break

        # Weather
        for cand in ["weather_condition", "weather"]:
            if cand in self.df.columns:
                rename_map[cand] = m.weather
                break

        # Promo
        for cand in ["holiday_promotion", "promotion", "promo", "holiday_promotion_"]:
            if cand in self.df.columns:
                rename_map[cand] = m.promo
                break

        # Competitor pricing
        for cand in ["competitor_pricing", "competitor_price"]:
            if cand in self.df.columns:
                rename_map[cand] = m.competitor
                break

        if rename_map:
            self.df.rename(columns=rename_map, inplace=True)

        # Fill required basics if missing
        for req in [m.product_id, m.category, m.region, m.inventory]:
            if req not in self.df.columns:
                raise ValueError(f"Required column '{req}' not found in inventory CSV!")

        # Optional numeric defaults
        for numc in [m.sold, m.forecast, m.price, m.discount]:
            if numc not in self.df.columns:
                self.df[numc] = 0

        # Strings
        for strc in [m.weather, m.promo, m.competitor]:
            if strc not in self.df.columns:
                self.df[strc] = ""

        # Ensure types
        for col in [m.inventory, m.sold, m.forecast, m.price, m.discount]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

        for col in [m.product_id, m.category, m.region, m.weather, m.promo, m.competitor]:
            self.df[col] = self.df[col].astype(str)

    def _ensure_text_feature(self) -> None:
        m = self.cols
        if m.text_feature not in self.df.columns:
            self.df[m.text_feature] = (
                self.df[m.category].fillna("")
                + " "
                + self.df[m.region].fillna("")
                + " "
                + self.df[m.weather].fillna("")
                + " "
                + self.df[m.promo].fillna("")
                + " "
                + self.df[m.competitor].fillna("")
            ).str.replace(r"\s+", " ", regex=True).str.strip()

    def _build_tfidf(self) -> None:
        m = self.cols
        self.vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
        self.tfidf = self.vectorizer.fit_transform(self.df[m.text_feature].fillna(""))

        # convenient ID -> index map
        self.id2idx = {
            pid: idx for idx, pid in enumerate(self.df[m.product_id].values)
        }

    # -----------------------------
    # Public methods used by API
    # -----------------------------
    def preview(self, limit: int = 15) -> List[Dict]:
        return self.df.head(int(limit)).to_dict(orient="records")

    def search(self, keyword: str, limit: int = 20) -> List[Dict]:
        if not keyword:
            return []
        mask = self.df[self.cols.text_feature].str.contains(
            str(keyword), case=False, na=False
        )
        return self.df.loc[mask].head(int(limit)).to_dict(orient="records")

    def recommendations(self, product_id: str, top_k: int = 5) -> List[Dict]:
        m = self.cols
        if product_id not in self.id2idx:
            return []
        idx = self.id2idx[product_id]
        sims = cosine_similarity(self.tfidf[idx], self.tfidf).ravel()
        order = np.argsort(-sims)

        out = []
        for j in order:
            if j == idx:
                continue
            row = self.df.iloc[j]
            out.append(
                {
                    "product_id": row[m.product_id],
                    "category": row[m.category],
                    "region": row[m.region],
                    "deadstock": bool(row.get("deadstock_flag", False)),
                    "similarity": float(sims[j]),
                    "distance": float(1.0 - sims[j]),
                }
            )
            if len(out) >= int(top_k):
                break
        return out

    def region_gaps(
        self, product_id: str, category_aware: bool = True
    ) -> pd.DataFrame:
        """
        Build a simple demand gap per region for the product's category (or all).
        gap = demand_proxy - supply_proxy

        - supply_proxy = sum inventory_level
        - demand_proxy = sum (units_sold + demand_forecast)
        """
        m = self.cols
        if product_id not in self.id2idx:
            return pd.DataFrame(columns=[m.region, "supply", "demand", "gap"])

        row = self.df.iloc[self.id2idx[product_id]]
        cat = row[m.category]

        df = self.df.copy()
        if category_aware:
            df = df[df[m.category] == cat]

        grp = df.groupby(m.region, as_index=False).agg(
            supply=(m.inventory, "sum"),
            demand=((m.sold, lambda s: s.sum()) if m.forecast not in df.columns
                    else (m.sold, "sum"))
        )
        # If forecast exists, add it
        if m.forecast in df.columns:
            add = df.groupby(m.region, as_index=False)[m.forecast].sum().rename(
                columns={m.forecast: "forecast"}
            )
            grp = grp.merge(add, on=m.region, how="left")
            grp["demand"] = grp["demand"].fillna(0) + grp["forecast"].fillna(0)

        grp["gap"] = grp["demand"].fillna(0) - grp["supply"].fillna(0)
        return grp[[m.region, "supply", "demand", "gap"]]

    def redistribution_plan(
        self,
        product_id: str,
        top_regions: int = 2,
        max_fraction: float = 0.2,
        tolerance_ratio: float = 0.0,
        category_aware: bool = True,
    ) -> Dict:
        """
        Heuristic: find regions with largest positive gap and allocate from the
        source region of the product subject to a max transfer fraction.
        """
        m = self.cols
        if product_id not in self.id2idx:
            return {"plan": [], "source_region": None, "available_qty": 0}

        row = self.df.iloc[self.id2idx[product_id]]
        source_region = row[m.region]
        category = row[m.category]

        # Available inventory at source (for the product's category)
        mask_src = (self.df[m.region] == source_region)
        if category_aware:
            mask_src &= (self.df[m.category] == category)
        available_qty = int(self.df.loc[mask_src, m.inventory].sum())

        budget = max(0, int(np.floor(available_qty * float(max_fraction))))

        gaps = self.region_gaps(product_id, category_aware=category_aware)
        # tolerance threshold (absolute)
        tol = float(tolerance_ratio) * max(1.0, gaps["demand"].abs().max())
        targets = gaps[gaps["gap"] > tol].sort_values("gap", ascending=False)

        out = []
        remain = budget
        for _, r in targets.iterrows():
            if remain <= 0:
                break
            if str(r[m.region]) == str(source_region):
                continue

            # allocate min of gap and remaining budget / regions left
            want = int(np.ceil(r["gap"]))
            if want <= 0:
                continue
            qty = min(remain, want)
            if qty <= 0:
                continue

            out.append(
                {
                    "target_region": r[m.region],
                    "suggested_qty": int(qty),
                    "gap": float(r["gap"]),
                }
            )
            remain -= qty
            if len(out) >= int(top_regions):
                break

        return {
            "source_region": str(source_region),
            "category": str(category),
            "available_qty": int(available_qty),
            "max_transfer": int(budget),
            "tolerance_used": float(tol),
            "plan": out,
            "regional_gaps": gaps.sort_values("gap", ascending=False).to_dict(orient="records"),
        }

    def partners(self, region: Optional[str] = None) -> List[Dict]:
        if self.partners_df.empty:
            return []
        df = self.partners_df.copy()
        # Normalize partner columns
        cols = {c.lower().strip(): c for c in df.columns}
        region_col = next((c for c in df.columns if c.lower() in {"region"}), None)
        ngo_col = next((c for c in df.columns if c.lower() in {"ngo", "partner", "name"}), None)
        contact_col = next((c for c in df.columns if "contact" in c.lower()), None)

        if region is not None and region_col:
            df = df[df[region_col].astype(str).str.lower() == str(region).lower()]

        df = df.rename(
            columns={
                region_col or "region": "region",
                ngo_col or "ngo": "ngo",
                contact_col or "contact": "contact",
            }
        )

        return df[["region", "ngo", "contact"]].to_dict(orient="records")

    def log_donation(
        self, product_id: str, qty: int, region: str, partner_name: str, contact: str, notes: str = ""
    ) -> Dict:
        """
        Append one donation record to donations_log.csv.
        On Render free tier this is ephemeral (resets on deploy).
        """
        ts = pd.Timestamp.utcnow().isoformat()
        row = {
            "timestamp": ts,
            "product_id": str(product_id),
            "qty": int(qty),
            "region": str(region),
            "partner_name": str(partner_name),
            "contact": str(contact),
            "notes": str(notes or ""),
        }
        try:
            pd.DataFrame([row]).to_csv(self.donations_log, mode="a", header=not self.donations_log.exists(), index=False)
        except Exception:
            # fallback: try reading to know if header exists
            if not self.donations_log.exists():
                pd.DataFrame(columns=list(row.keys())).to_csv(self.donations_log, index=False)
            pd.DataFrame([row]).to_csv(self.donations_log, mode="a", header=False, index=False)
        return row
