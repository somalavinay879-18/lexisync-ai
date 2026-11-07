FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Ollama runs on the host machine: docker run -p 8501:8501 lexisync
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    PYTHONHASHSEED=42
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
