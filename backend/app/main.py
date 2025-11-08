"""
FastAPI app for Deadstock Dilemma.

Endpoints:
- GET /               -> health banner for frontend ping
- GET /health         -> status ok
- GET /inventory      -> preview of inventory rows
- GET /search         -> keyword search
- GET /recommendations -> similar items by product_id
- GET /redistribution_plan -> suggested flows
- GET /partners       -> NGO list (optional file)
- POST /donate        -> log a donation to CSV
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr

from .recommender import DeadstockRecommender

app = FastAPI(title="Deadstock Dilemma AI Backend", version="1.4.0")

# CORS: keep open for initial deployment; later restrict to your Streamlit URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://deadstock-frontend.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init single instance
recommender = DeadstockRecommender()

# -------- Meta ----------
@app.get("/", tags=["meta"])
def root():
    return {"message": "🧠 Deadstock Dilemma AI Backend is running!"}

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}

# -------- Inventory ----------
@app.get("/inventory", tags=["inventory"])
def inventory(limit: int = Query(15, ge=1, le=200)):
    try:
        return recommender.preview(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Search ----------
@app.get("/search", tags=["search"])
def search(keyword: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=200)):
    try:
        return recommender.search(keyword=keyword, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Recommendations ----------
@app.get("/recommendations", tags=["recommend"])
def recommendations(product_id: str = Query(...), top_k: int = Query(5, ge=1, le=50)):
    try:
        items = recommender.recommendations(product_id=product_id, top_k=top_k)
        if not items:
            raise HTTPException(status_code=404, detail=f"No recommendations for product_id={product_id}")
        return items
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Redistribution ----------
@app.get("/redistribution_plan", tags=["redistribution"])
def redistribution_plan(
    product_id: str = Query(...),
    top_regions: int = Query(2, ge=1, le=10),
    max_fraction: float = Query(0.2, ge=0.0, le=1.0),
    tolerance_ratio: float = Query(0.0, ge=0.0, le=1.0),
    category_aware: bool = Query(True),
):
    try:
        return recommender.redistribution_plan(
            product_id=product_id,
            top_regions=top_regions,
            max_fraction=max_fraction,
            tolerance_ratio=tolerance_ratio,
            category_aware=category_aware,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Partners ----------
@app.get("/partners", tags=["partners"])
def partners(region: Optional[str] = None):
    try:
        return recommender.partners(region=region)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Donation ----------
class DonationRequest(BaseModel):
    product_id: str
    qty: int = Field(ge=1)
    region: str
    partner_name: str
    contact: str  # keep free-form for now; switch to EmailStr if you want strict emails
    notes: Optional[str] = None

@app.post("/donate", tags=["partners"])
def donate(payload: DonationRequest):
    try:
        logged = recommender.log_donation(
            product_id=payload.product_id,
            qty=int(payload.qty),
            region=payload.region,
            partner_name=payload.partner_name,
            contact=payload.contact,
            notes=payload.notes or "",
        )
        return {"ok": True, "record": logged}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
