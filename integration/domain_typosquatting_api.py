from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# =========================================================
# IMPORT EXISTING FINAL DETECTOR
# =========================================================

from src.final_domain_typosquatting_detector import (
    predict_url,
)


# =========================================================
# CREATE API
# =========================================================

app = FastAPI(
    title="Domain & Typosquatting Detection API",
    description=(
        "Final offline domain and typosquatting phishing "
        "detection component."
    ),
    version="1.0"
)


# =========================================================
# INPUT FORMAT
# =========================================================

class URLRequest(BaseModel):
    url: str


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "component": "domain_and_typosquatting",
        "status": "online"
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
def predict(request: URLRequest):

    try:

        result = predict_url(
            request.url
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


