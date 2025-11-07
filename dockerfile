FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el repo
COPY . .

# Expone el puerto de FastAPI
EXPOSE 7860

# Comando de ejecución
CMD ["bash", "app.sh"]
