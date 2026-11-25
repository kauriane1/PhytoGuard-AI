# PhytoGuard AI - API container (FastAPI + TensorFlow model)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, trained model and sample data
COPY src/ src/
COPY models/ models/
COPY data/ data/

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# PORT is set by the cloud platform (Render) and defaults to 8000 locally
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT}
