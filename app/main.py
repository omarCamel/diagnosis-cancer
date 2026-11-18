import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.database import get_predictions, initialize_database, save_prediction
from app.schemas import CustomerInput, PredictionHistoryItem, PredictionResult

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data.csv"
model = joblib.load(BASE_DIR / "models" / "champion.joblib")
dataset = pd.read_csv(DATASET_PATH).dropna(axis=1, how="all")


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(title="API de diagnóstico de cáncer", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/dataset")
@app.get("/data")
async def dataset_summary():
    cleaned = dataset.dropna(axis=1, how="all").copy()
    preview_df = cleaned.head(5).copy()
    preview_df = preview_df.where(pd.notna(preview_df), None)
    preview = preview_df.to_dict(orient="records")
    return {
        "rows": int(cleaned.shape[0]),
        "columns": int(cleaned.shape[1]),
        "column_names": list(cleaned.columns),
        "preview": preview,
    }


@app.get("/", response_class=HTMLResponse)
async def home():
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.post("/predict", response_model=PredictionResult)
def predict(customer: CustomerInput):
    input_features = customer.model_dump(by_alias=True)
    frame = pd.DataFrame([input_features])
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    risk = "Bajo" if probability < 0.30 else "Medio" if probability < 0.70 else "Alto"
    diagnosis_label = "Maligno" if prediction else "Benigno"
    result = {
        "prediction": prediction,
        "diagnosis_probability": probability,
        "risk_level": risk,
        "message": f"Diagnóstico {diagnosis_label}: probabilidad estimada de malignidad {probability:.3f}",
    }
    save_prediction(input_features, result)
    return result


def require_history_api_key(
    x_history_api_key: str | None = Header(default=None, alias="X-History-API-Key"),
) -> None:
    expected_api_key = os.getenv("PREDICTIONS_HISTORY_API_KEY", "")
    if not expected_api_key.strip():
        raise HTTPException(
            status_code=503,
            detail="El acceso al historial no está configurado.",
        )
    if not x_history_api_key or not secrets.compare_digest(
        x_history_api_key.encode("utf-8"), expected_api_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="API key inválida o ausente.")


@app.get("/predictions", response_model=list[PredictionHistoryItem])
def prediction_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _: None = Depends(require_history_api_key),
):
    return get_predictions(limit=limit, offset=offset)
