import os
import pandas as pd
import joblib
from typing import List, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# =========================
# Config del modelo en HF Hub
# =========================
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "ramen159/model-agente-continuidad")
HF_MODEL_FILE = os.getenv("HF_MODEL_FILE", "modelo_desercion.pkl")

print(f"[INFO] Descargando modelo desde HuggingFace Hub: {HF_MODEL_REPO}/{HF_MODEL_FILE}")
try:
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILE,
        force_download=False  # usa caché si ya existe
    )
    MODEL = joblib.load(model_path)
    print("[INFO] Modelo cargado correctamente ✅")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    MODEL = None

# =========================
# Schemas de entrada
# =========================
class ChatBody(BaseModel):
    messages: List[Dict[str, str]]

class PredictBody(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]

# =========================
# API FastAPI
# =========================
app = FastAPI(title="API Continuidad Académica")

@app.post("/chat")
def chat(body: ChatBody):
    last_user_msg = next((m["content"] for m in reversed(body.messages) if m["role"] == "user"), "")
    if not last_user_msg:
        return {"reply": "¿En qué te puedo ayudar?"}
    return {"reply": f"Recibí tu mensaje: {last_user_msg}"}

@app.post("/predict")
def predict(body: PredictBody):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado en memoria.")

    rows = body.data if isinstance(body.data, list) else [body.data]

    try:
        df = pd.DataFrame(rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Formato de entrada inválido: {e}")

    try:
        preds = MODEL.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    preds = [float(p) if hasattr(p, "item") else p for p in preds]
    return {"predictions": preds}
