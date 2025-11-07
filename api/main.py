import os
from typing import List, Dict, Any, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

# =========================
# Config HF Hub
# =========================
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "ramen159/model-agente-continuidad")
HF_MODEL_FILE = os.getenv("HF_MODEL_FILE", "modelo_desercion.pkl")
HF_ENCODER_FILE = os.getenv("HF_ENCODER_FILE", "encoder.pkl")

# =========================
# Columnas reales del encoder
# (detectadas desde tu dataset)
# =========================
ENCODER_INPUT_FEATURES = [
    "AGNO","NOM_RBD","NOM_REG_RBD_A","NOM_COM_RBD","NOM_DEPROV_RBD",
    "COD_DEPE","COD_DEPE2","RURAL_RBD","ESTADO_ESTAB","COD_ENSE",
    "COD_GRADO","COD_JOR","MRUN","GEN_ALU","FEC_NAC_ALU",
    "NOM_COM_ALU","SIT_FIN","SIT_FIN_R"
]
TARGET_COL = "Desercion"

# =========================
# Carga desde Hugging Face
# =========================
def load_pkl(repo: str, filename: str):
    path = hf_hub_download(repo_id=repo, filename=filename, force_download=False)
    return joblib.load(path)

MODEL = None
ENCODER = None

print(f"[INFO] Cargando modelo: {HF_MODEL_REPO}/{HF_MODEL_FILE}")
try:
    MODEL = load_pkl(HF_MODEL_REPO, HF_MODEL_FILE)
    print("[INFO] Modelo cargado ✅")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")

print(f"[INFO] Cargando encoder: {HF_MODEL_REPO}/{HF_ENCODER_FILE}")
try:
    ENCODER = load_pkl(HF_MODEL_REPO, HF_ENCODER_FILE)
    print("[INFO] Encoder cargado ✅")
except Exception as e:
    print(f"[WARN] No se pudo cargar el encoder: {e}")
    ENCODER = None

# Después de cargar ENCODER
try:
    # Fuerza a usar EXACTAMENTE las columnas que el OrdinalEncoder vio en el fit
    ENCODER_INPUT_FEATURES = list(ENCODER.feature_names_in_)
except Exception:
    # Si por alguna razón no expone feature_names_in_, usa la lista manual (de tu debug)
    ENCODER_INPUT_FEATURES = [
        "COD_ENSE","NOM_RBD","COD_JOR","COD_GRADO",
        "NOM_COM_RBD","NOM_COM_ALU","NOM_DEPROV_RBD",
        "NOM_REG_RBD_A","COD_DEPE"
    ]

print("[INFO] ENCODER_INPUT_FEATURES:", ENCODER_INPUT_FEATURES)

# =========================
# Utilidad para detectar columnas del modelo
# =========================
def get_model_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

MODEL_FEATURES = get_model_feature_names(MODEL)
print("[INFO] MODEL_FEATURES:", MODEL_FEATURES)

# =========================
# FastAPI Schemas
# =========================
class ChatBody(BaseModel):
    messages: List[Dict[str, str]]

class PredictBody(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]

# =========================
# FastAPI App
# =========================
app = FastAPI(title="API Continuidad Académica")

@app.get("/schema")
def schema():
    return {
        "model_repo": HF_MODEL_REPO,
        "model_file": HF_MODEL_FILE,
        "encoder_file": HF_ENCODER_FILE,
        "encoder_input_features": ENCODER_INPUT_FEATURES,
        "model_feature_names": MODEL_FEATURES,
        "nota": f"No enviar '{TARGET_COL}' en /predict"
    }

@app.post("/chat")
def chat(body: ChatBody):
    last = next((m["content"] for m in reversed(body.messages) if m["role"] == "user"), None)
    return {"reply": "¿En qué te puedo ayudar?" if not last else f"Recibí tu mensaje: {last}"}

# =========================
# Predicción
# =========================
@app.post("/predict")
def predict(body: PredictBody):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    rows = body.data if isinstance(body.data, list) else [body.data]
    try:
        df = pd.DataFrame(rows)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON inválido: {e}")

    # Nunca mandar el target
    for tgt in ("Desercion", "desercion", "target"):
        if tgt in df.columns:
            df = df.drop(columns=[tgt])

    # --- Descubrir columnas que espera el encoder (las que vimos en tu debug)
    expected = ENCODER_INPUT_FEATURES or (
        list(ENCODER.feature_names_in_) if hasattr(ENCODER, "feature_names_in_") else None
    )
    if not expected:
        raise HTTPException(status_code=500, detail="No se pudo determinar ENCODER_INPUT_FEATURES.")

    # --- Diagnóstico explícito
    incoming = list(df.columns)
    missing   = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in expected]

    if missing or unexpected:
        return {
            "detail": "Las columnas no coinciden con las del encoder.",
            "expected": expected,
            "incoming": incoming,
            "missing": missing,
            "unexpected": unexpected
        }, 400

    # --- Recortar y ordenar EXACTAMENTE en el orden del encoder
    df = df[expected]

    # --- OrdinalEncoder fue fit con dtype 'object' (strings) → castear todo a str
    for c in expected:
        df[c] = df[c].astype(str)

    # (Opcional) muestra qué columnas se enviarán al encoder
    # print("[DEBUG] columns to encoder:", list(df.columns))

    # --- Transformar con encoder
    try:
        X = ENCODER.transform(df) if ENCODER is not None else df.copy()
        if hasattr(X, "toarray"):  # sparse
            X = X.toarray()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al transformar con encoder: {e}")

    # --- Alinear a lo que el MODELO espera (si lo expone)
    if MODEL_FEATURES is not None:
        for col in MODEL_FEATURES:
            if col not in X.columns:
                X[col] = 0
        X = X[MODEL_FEATURES]

    # --- Predecir
    try:
        preds = MODEL.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

    preds = [float(p) if hasattr(p, "item") else p for p in preds]
    return {"predictions": preds, "rows": len(preds)}