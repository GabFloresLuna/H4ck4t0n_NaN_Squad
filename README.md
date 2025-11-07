# 🧠 Tutor Virtual de Continuidad Académica  
*Hackathon Duoc UC 2025 — Equipo H4ck4t0n_NaN_Squad*

Este proyecto implementa un **sistema inteligente híbrido** que combina:

| Componente | Tecnología |
|------------|------------|
| Modelo de riesgo de deserción | Scikit-Learn (cargado desde HuggingFace Hub) |
| API backend | FastAPI + Uvicorn |
| Motor de recomendaciones | RAG basado en Markdown local |
| Interfaz web | Gradio (integrada en el mismo contenedor) |
| Despliegue | Hugging Face Spaces (Docker) |

---
## Estructura del repo
.
├── api/
│   └── main.py           # Backend FastAPI + RAG + UI
├── kb/                   # Markdown con recomendaciones
│   ├── habitos_estudio.md
│   └── salud_mental.md
├── app.sh                # Comando de ejecución (opcional)
├── Dockerfile            # Space basado en Docker
├── requirements.txt
└── README.md


## 🎯 Objetivo

1. **Predecir el riesgo académico** (`0 = sin riesgo`, `1 = riesgo alto`)
2. **Explicar y acompañar al estudiante** si hay riesgo → entregando **recomendaciones personalizadas**
3. **RAG liviano offline**: extrae tips desde archivos `.md` ubicados en `/kb`

---

## 🧩 Arquitectura


## H4ck4t0n_NaN_Squad – Hackathon Duoc UC 2025
  - Cristopher Ormazabal
  - Cristobal Pardo
  - Dante Valle
  - Gabriel Flores

┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│ FastAPI │ --> │ Modelo ML │ --> │ Predicción │
└─────┬───────┘ └──────────────┘ └──────┬───────┘
│ (si = 1)
▼
┌──────────────┐ RAG Markdown ┌─────────────────────────┐
│ Recomendador│ <────────────── │ /kb/*.md (tips + guías) │
└──────────────┘ └─────────────────────────┘



## 📌 JSON de ejemplo válido (9 features requeridos)

```json
{
  "data": {
    "COD_ENSE": "110",
    "NOM_RBD": "LICEO EJEMPLO",
    "COD_JOR": "1",
    "COD_GRADO": "7",
    "NOM_COM_RBD": "SANTIAGO",
    "NOM_COM_ALU": "SANTIAGO",
    "NOM_DEPROV_RBD": "SANTIAGO",
    "NOM_REG_RBD_A": "REGIÓN METROPOLITANA",
    "COD_DEPE": "Municipal DAEM"
  }
}


