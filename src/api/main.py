from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from src.api.coach import router as coach_router
from src.api.model_info import router as model_info_router
from src.rag.query_api import search_papers

app = FastAPI(title="SoleusAI")

model = joblib.load("models/emg_activation_model.pkl")
app.include_router(coach_router)
app.include_router(model_info_router)
class EMGInput(BaseModel):
    rms: float
    mav: float
    zcr: int
    wl: float


class QuestionInput(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "SoleusAI Running"}


@app.post("/predict")
def predict(data: EMGInput):

    sample = pd.DataFrame([{
        "rms": data.rms,
        "mav": data.mav,
        "zcr": data.zcr,
        "wl": data.wl
    }])

    prediction = model.predict(sample)

    return {
        "activation_score": round(float(prediction[0]), 2)
    }


@app.post("/ask")
def ask(data: QuestionInput):

    answer = search_papers(data.question)

    return {
        "answer": answer
    }