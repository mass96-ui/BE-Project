from fastapi import APIRouter
import pandas as pd

router = APIRouter()

@router.get("/model-info")
def model_info():

    df = pd.read_csv("models/model_comparison.csv")

    best = df.sort_values(
        by="R2",
        ascending=False
    ).iloc[0]

    return {
        "project": "SoleusAI",
        "best_model": best["Model"],
        "r2_score": round(float(best["R2"]), 4),
        "features": [
            "rms",
            "mav",
            "zcr",
            "wl"
        ],
        "capabilities": [
            "EMG Feature Extraction",
            "Activation Prediction",
            "AI Coach",
            "Research Assistant"
        ]
    }