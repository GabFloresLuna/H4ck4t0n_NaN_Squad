import os
import joblib
import pandas as pd
from typing import List, Dict, Any, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =========================
# Cargar modelo sklearn (.pkl o .joblib)
# =========================
MODEL = None
for cand in ["model.pkl", "model.joblib", "models/model.pkl", "models/model.joblib"]:
    if os.path.exists(cand):
        try:
            MODEL = joblib.load(cand)
            print(f"[INFO] Modelo cargado desde: {cand}")
            break
        except Exception as e:
            print(f"[WARN] No se pudo cargar {cand}: {e}")

# =========================
# Definir estructura de input
# =========================
class ChatBody(BaseModel):
    messages: List[Dict[str, str]]

class PredictBody(BaseModel):
    # Puede venir un diccionario o una lista de diccionarios
    data: Union[Dict[str, Any], List[Dict[str, Any]]]

# =========================
# Crear la API
# =========================
app = FastAPI(title="API Chat + Predict")

@app.post("/chat")
def chat(body: ChatBody):
    """
    Chat mínimo: responde con eco/placeholder.
    Luego puedes reemplazarlo con tu propia lógica (por ejemplo, LLM).
    """
    last_user_message = next(
        (m["content"] for m in reversed(body.messages) if m["role"] == "user"),
        ""
    )
    if not last_user_message:
        return {"reply": "¿En qué te ayudo?"}
    return {"reply": f"Recibí tu mensaje: {last_user_message}"}

@app.post("/predict")
def predict(body: PredictBody):
    if MODEL is None:
        raise HTTPException(status_code=501, detail="No se encontró el modelo 'model.pkl' o 'model.joblib'.")

    rows = body.data if isinstance(body.data, list) else [body.data]
    try:
        df = pd.DataFrame(rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al convertir entrada a DataFrame: {e}")

    try:
        preds = MODEL.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {e}")

    preds = [float(p) if hasattr(p, "item") else p for p in preds]
    return {"predictions": preds}
