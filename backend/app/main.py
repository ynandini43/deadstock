from fastapi import FastAPI, Query
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from app.recommender import DeadstockRecommender
from app.partners import partners_by_region, append_donation_record

app = FastAPI(title="Deadstock Dilemma AI Backend", version="1.4.0")
recommender = DeadstockRecommender()

@app.get("/")
def root():
    return {"message": "🧠 Deadstock Dilemma AI Backend is running!"}

@app.get("/inventory")
def get_inventory(limit: int = 10):
    return recommender.get_inventory(limit=limit)

@app.get("/search")
def search_items(keyword: str):
    return {"query": keyword, "results": recommender.search(keyword)}

@app.get("/recommendations")
def get_recommendations(product_id: str = Query(...), top_k: int = 5):
    return {"product_id": product_id, "recommendations": recommender.recommend_for_product_id(product_id, top_k)}

@app.get("/redistribution_plan")
def redistribution_plan(
    product_id: str = Query(...),
    top_regions: int = 3,
    max_fraction: float = 0.5,
    category_aware: bool = True,
    tolerance_ratio: float = 0.10,
):
    return recommender.redistribution_plan_for_product(
        product_id,
        top_regions=top_regions,
        max_fraction=max_fraction,
        category_aware=category_aware,
        tolerance_ratio=tolerance_ratio,
    )

# ---------- NEW: partners & donation ----------

@app.get("/partners")
def list_partners(region: Optional[str] = None):
    """Return NGO partners; filtered by region if provided."""
    return {"region": region, "partners": partners_by_region(region)}

class DonationRequest(BaseModel):
    product_id: str = Field(..., description="Product to donate")
    qty: int = Field(..., ge=1)
    region: str
    partner_ngo: str
    partner_contact: Optional[str] = None
    partner_email: Optional[EmailStr] = None
    notes: Optional[str] = None

@app.post("/donate")
def donate(req: DonationRequest):
    """Append a donation record to backend/data/donations_log.csv"""
    record = req.dict()
    append_donation_record(record)
    return {"status": "ok", "recorded": record}
