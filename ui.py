import os
import json
import requests
import gradio as gr

API_URL = os.getenv("API_URL", "http://localhost:7860/predict")

EXAMPLE = {
    "data": {
        "COD_ENSE": "110",
        "NOM_RBD": "LICEO EJEMPLO",
        "COD_GRADO": "7"
    }
}

def call_api(json_text):
    try:
        payload = json.loads(json_text)
        if "data" not in payload:
            payload = {"data": payload}
    except Exception as e:
        return f"JSON inválido: {e}"

    try:
        res = requests.post(API_URL, json=payload)
        return res.json()
    except Exception as e:
        return f"Error al conectar API: {e}"

def build_demo():
    with gr.Blocks(title="Agente Continuidad Académica") as demo:
        gr.Markdown("## Demo API + UI")
        inp = gr.Textbox(value=json.dumps(EXAMPLE, indent=2), lines=12, label="Entrada JSON")
        out = gr.JSON(label="Respuesta")
        gr.Button("Predecir").click(call_api, inp, out)
        gr.Markdown("### Endpoints\n- `/predict`\n- `/docs`")
    return demo

