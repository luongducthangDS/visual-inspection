FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vqi/ vqi/
# Weights come from training (python -m vqi.classify ..., python -m vqi.detect, python -m vqi.report); mounted at runtime.
ENV ARTIFACTS_DIR=/app/artifacts INSPECTION_DIR=/app/inspections YOLO_CONFIG_DIR=/tmp
EXPOSE 8002
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8002/health')"
CMD ["uvicorn", "vqi.api:app", "--host", "0.0.0.0", "--port", "8002"]
