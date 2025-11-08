# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .recommender import DeadstockRecommender
# from .routes import router   # if you use APIRouter for endpoints

app = FastAPI(title="Deadstock Dilemma AI Backend", version="1.4.0")

# CORS: for the first deploy keep it open; later lock to your Streamlit URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later: ["https://<your-frontend>.onrender.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Health / Root endpoints (fixes 404 on Render root) ----
@app.get("/", tags=["meta"])
def root():
    return {"message": "🧠 Deadstock Dilemma AI Backend is running!"}

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}

# Optional: redirect /api -> /
@app.get("/api", include_in_schema=False)
def api_root_redirect():
    return RedirectResponse(url="/")

# ---- If you use APIRouter with a prefix, mount both ways ----
# app.include_router(router)            # no prefix
# app.include_router(router, prefix="/api")  # with /api
# -------------------------------------------------------------
