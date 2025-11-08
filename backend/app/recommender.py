import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


class DeadstockRecommender:
    """
    - Loads/creates processed_inventory.csv
    - Search + content-based recommendations
    - Redistribution planner:
        * category-aware regional balancing
        * sell-through driven gap (robust when dataset has no true shortages)
        * tolerance to allow near-balanced regions
        * fallback so you always get a plan
    """

    def __init__(self, processed_csv: str = "backend/data/processed_inventory.csv"):
        self.processed_path = self._ensure_processed_csv(processed_csv)
        print(f"📂 Loading data from: {self.processed_path}")
        self.df = pd.read_csv(self.processed_path)

        # normalize columns
        self.df.columns = [c.strip().replace(" ", "_").lower() for c in self.df.columns]
        required = {
            "product_id",
            "category",
            "region",
            "inventory_level",
            "units_sold",
            "deadstock_flag",
            "text_feature",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise KeyError(f"Processed CSV missing columns: {missing}")

        # embeddings & index for recommendations/search
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.embeddings = self.vectorizer.fit_transform(self.df["text_feature"].fillna(""))
        self.nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(self.embeddings)
        print(f"✅ Loaded {len(self.df)} inventory items successfully.")

    # --------------------------- Public helpers ---------------------------

    def get_inventory(self, limit=10):
        return self.df.head(limit).to_dict(orient="records")

    def search(self, keyword: str):
        mask = self.df["text_feature"].str.contains(keyword, case=False, na=False)
        return self.df[mask].head(10).to_dict(orient="records")

    def recommend_for_product_id(self, pid: str, top_k: int = 5):
        idx = self._index_for_pid(pid)
        if idx is None:
            return []
        return self._recommend_by_index(idx, top_k=top_k)

    # --------------------------- Recommendations --------------------------

    def _recommend_by_index(self, idx: int, top_k: int = 5):
        q = self.embeddings[idx]
        n = min(top_k + 1, len(self.df))
        dists, idxs = self.nn.kneighbors(q, n_neighbors=n)
        results = []
        for dist, i in zip(dists[0], idxs[0]):
            if i == idx:
                continue  # skip self
            row = self.df.iloc[i]
            results.append(
                {
                    "Product_ID": row.get("product_id"),
                    "Category": row.get("category"),
                    "Region": row.get("region"),
                    "Distance": round(float(dist), 3),
                    "Deadstock": bool(row.get("deadstock_flag", False)),
                }
            )
        return results

    # --------------------------- Redistribution --------------------------

    def redistribution_plan_for_product(
        self,
        pid: str,
        top_regions: int = 3,
        max_fraction: float = 0.5,
        category_aware: bool = True,
        tolerance_ratio: float = 0.10,
    ):
        """
        Robust redistribution when true shortages don't exist:

        1) Build sell-through = (units_sold+1) / (inventory_level+1)
        2) Within product category (or globally), find regions with LOW relative sell-through.
           Define a pseudo 'gap' that is positive when the region under-performs vs the average.
        3) Allow tolerance so near-balanced regions still qualify.
        4) If nothing qualifies, fallback to lowest sell-through regions.
        5) Suggest quantities capped by max_fraction of available stock, and score with similarity.
        """
        idx = self._index_for_pid(pid)
        if idx is None:
            return {"error": "product_id not found", "plan": []}

        item = self.df.iloc[idx]
        source_region = item["region"]
        category = item["category"]
        available_qty = int(item["inventory_level"] or 0)
        if available_qty <= 0:
            return {
                "product_id": pid,
                "source_region": source_region,
                "category": category,
                "available_qty": 0,
                "max_transfer_considered": 0,
                "tolerance_used": 0.0,
                "plan": [],
                "reason_if_empty": "No available inventory to move.",
                "gap_preview": [],
            }

        # Build sell-through
        df = self.df.copy()
        df["sell_ratio"] = (df["units_sold"].astype(float) + 1.0) / (
            df["inventory_level"].astype(float) + 1.0
        )
        avg_sell_ratio = df["sell_ratio"].mean()

        # Category-aware pool (preferred)
        if category_aware:
            pool = df[(df["category"] == category) & (df["region"] != source_region)]
        else:
            pool = df[df["region"] != source_region]

        # Aggregate by region
        region_stats = (
            pool.groupby("region", dropna=False)
            .agg(
                inventory_level=("inventory_level", "sum"),
                units_sold=("units_sold", "sum"),
                sell_ratio=("sell_ratio", "mean"),
                items=("product_id", "count"),
            )
            .reset_index()
        )

        # Pseudo demand gap: regions below avg sell-through get positive "need"
        region_stats["gap"] = (avg_sell_ratio - region_stats["sell_ratio"]) * region_stats[
            "inventory_level"
        ]
        region_stats["gap"] = region_stats["gap"].fillna(0.0).astype(float)

        # Tolerance (let near-balanced regions qualify)
        base = region_stats["gap"].abs().mean() if len(region_stats) else 0.0
        tol = max(0.0, tolerance_ratio * (base if base > 0 else 1.0))
        region_stats["eligible"] = region_stats["gap"] > -tol

        # Choose targets
        targets = (
            region_stats[region_stats["eligible"]]
            .sort_values("gap", ascending=False)
            .head(top_regions)
            .copy()
        )

        fallback_used = False
        reason = "Selected regions have lower relative sell-through and suit rebalancing."
        if targets.empty:
            # Fallback: pick lowest sell-through regions globally (still excluding source)
            global_stats = (
                df[df["region"] != source_region]
                .groupby("region", dropna=False)
                .agg(
                    inventory_level=("inventory_level", "sum"),
                    units_sold=("units_sold", "sum"),
                    sell_ratio=("sell_ratio", "mean"),
                    items=("product_id", "count"),
                )
                .reset_index()
            )
            targets = global_stats.sort_values("sell_ratio", ascending=True).head(top_regions)
            fallback_used = True
            reason = "No true demand regions found — proposing low sell-through regions for rebalancing."

        max_transfer = max(1, int(available_qty * float(max_fraction)))
        per_region = max(1, int(max_transfer / max(1, top_regions)))

        plan = []
        for _, trg in targets.iterrows():
            trg_region = trg["region"]
            # Similarity to items in target region (prefer same category)
            if category_aware:
                trg_idx = self.df.index[
                    (self.df["region"] == trg_region) & (self.df["category"] == category)
                ].tolist()
            else:
                trg_idx = self.df.index[self.df["region"] == trg_region].tolist()

            if trg_idx:
                sims = cosine_similarity(self.embeddings[idx], self.embeddings[trg_idx]).flatten()
                sim = float(sims.max())
            else:
                sim = 0.0

            gap_val = float(trg.get("gap", 0.0))
            plan.append(
                {
                    "target_region": str(trg_region),
                    "suggested_qty": int(per_region),
                    "gap": round(gap_val, 2),
                    "similarity": round(sim, 3),
                    "reason": f"{reason} (tol={tol:.2f}).",
                }
            )

        # For diagnostics
        gap_preview_cols = [
            c for c in ["region", "inventory_level", "units_sold", "sell_ratio", "gap"] if c in region_stats.columns
        ]
        gap_preview = region_stats[gap_preview_cols].to_dict(orient="records")

        return {
            "product_id": pid,
            "source_region": source_region,
            "category": category,
            "available_qty": int(available_qty),
            "max_transfer_considered": int(max_transfer),
            "tolerance_used": round(tol, 3),
            "category_aware": bool(category_aware),
            "fallback_to_global": bool(fallback_used),
            "gap_preview": gap_preview,
            "plan": plan,
            "reason_if_empty": None if plan else reason,
        }

    # --------------------------- Internal ---------------------------

    def _index_for_pid(self, pid: str):
        hits = self.df.index[self.df["product_id"].astype(str) == str(pid)].tolist()
        return hits[0] if hits else None

    def _ensure_processed_csv(self, processed_csv: str) -> Path:
        """Locate or build processed_inventory.csv from raw if needed."""
        cwd = Path().resolve()
        candidates = [
            cwd / processed_csv,
            cwd / "data" / "processed_inventory.csv",
            cwd.parent / "backend" / "data" / "processed_inventory.csv",
        ]
        for p in candidates:
            if p.exists():
                return p

        # try to build from raw
        raw_candidates = [
            cwd / "data" / "retail_inventory.csv",
            cwd.parent / "data" / "retail_inventory.csv",
            cwd / "backend" / "data" / "retail_inventory.csv",
        ]
        raw_path = next((p for p in raw_candidates if p.exists()), None)
        if raw_path is None:
            raise FileNotFoundError(
                "processed_inventory.csv not found and raw dataset missing.\n"
                f"Looked for processed at: {[str(p) for p in candidates]}\n"
                f"Looked for raw at: {[str(p) for p in raw_candidates]}"
            )

        out_dir = cwd / "backend" / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "processed_inventory.csv"

        df = pd.read_csv(raw_path)
        df.columns = [c.strip() for c in df.columns]
        if "Inventory Level" not in df.columns or "Units Sold" not in df.columns:
            raise KeyError("Raw CSV missing 'Inventory Level' or 'Units Sold'.")

        df["Inventory Level"] = pd.to_numeric(df["Inventory Level"], errors="coerce")
        df["Units Sold"] = pd.to_numeric(df["Units Sold"], errors="coerce")

        mean_inv = df["Inventory Level"].mean()
        mean_sales = df["Units Sold"].mean()
        df["deadstock_flag"] = (df["Inventory Level"] > mean_inv * 1.5) & (
            df["Units Sold"] < mean_sales * 0.5
        )

        # text_feature
        text_cols = [
            "Category",
            "Region",
            "Weather Condition",
            "Holiday/Promotion",
            "Competitor Pricing",
        ]
        for c in text_cols:
            if c not in df.columns:
                df[c] = ""
            df[c] = df[c].astype(str).fillna("")
        df["text_feature"] = (
            df["Category"]
            + " "
            + df["Region"]
            + " "
            + df["Weather Condition"]
            + " "
            + df["Holiday/Promotion"]
            + " "
            + df["Competitor Pricing"]
        ).str.strip()

        # rename
        rename = {
            "Product ID": "product_id",
            "Category": "category",
            "Region": "region",
            "Inventory Level": "inventory_level",
            "Units Sold": "units_sold",
        }
        df.rename(columns=rename, inplace=True)

        df.to_csv(out_path, index=False)
        print(f"🛠️ Built processed_inventory.csv from raw → {out_path}")
        return out_path
