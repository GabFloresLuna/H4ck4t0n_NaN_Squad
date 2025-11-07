## Hackathon Duoc UC 2025
Tutor Virtual Adaptativo con IA Híbrida para estimar riesgo de deserción y ofrecer planes personalizados.

## Objetivo
Desarrollar un sistema híbrido (ML + LLM + RAG) capaz de:
- Estimar riesgo de deserción escolar.
- Explicar las variables relevantes.
- Recomendar planes personalizados.

## Estructura del proyecto
## 🗂️ Estructura del proyecto

```text
education-hackathon-Duoc/
├── data/              # Datasets (rendimiento, asistencia, deserción)
├── kb/                # Base de conocimiento local (RAG)
├── src/               # Código ML, RAG, validadores
├── api/               # FastAPI endpoints (/predict, /coach)
├── app/               # App demo (Streamlit/Gradio)
├── requirements.txt   # Dependencias
└── README.md          # Documentación principal
```

## H4ck4t0n_NaN_Squad – Hackathon Duoc UC 2025
  - Cristopher Ormazabal
  - Cristobal Pardo
  - Dante Valle
  - Gabriel Flores
# 🎓 API Tutor Virtual - Hackathon Duoc UC 2025

## 🚀 Descripción
Esta API predice el **riesgo académico de los estudiantes** y entrega un **plan de acción personalizado (coach)**.  
Está desarrollada con **FastAPI** y se encuentra actualmente **operativa en la nube**.

---

## 🌐 Enlace Público
**Base URL:**  
https://bedroom-injection-winners-print.trycloudflare.com

**Documentación (Swagger UI):**  
https://bedroom-injection-winners-print.trycloudflare.com/docs

---

## 📈 Endpoint: `/predict`
**Método:** `POST`  
**Descripción:** Predice el riesgo académico según características del estudiante.

### Ejemplo de entrada:
```json
{
  "age": 17,
  "sex": "M",
  "school": "Liceo Técnico",
  "subject": "Matemáticas",
  "attendance_pct": 75,
  "grade_mean": 4.1,
  "num_absences": 10,
  "socioeconomic_status": "low"
}
```
