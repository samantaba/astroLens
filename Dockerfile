FROM python:3.11-slim

LABEL maintainer="Saman Tabatabaeian <saman.tabatabaeian@gmail.com>"
LABEL description="AstroLens - AI-Powered Galaxy Anomaly Discovery System"
LABEL version="1.0.0"

WORKDIR /app

# Install system dependencies (OpenCV, GL for image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY api/ ./api/
COPY inference/ ./inference/
COPY features/ ./features/
COPY catalog/ ./catalog/
COPY annotator/ ./annotator/
COPY agent/ ./agent/
COPY scripts/ ./scripts/
COPY transient_detector/ ./transient_detector/
COPY web/ ./web/
COPY finetuning/ ./finetuning/
COPY ingest/ ./ingest/
COPY paths.py ./paths.py

# Copy pre-trained models
COPY models/ ./models/

# Create data directories
RUN mkdir -p /data/images /data/downloads /data/exports \
    /data/transient_data /data/weights

# Expose ports: API (8000), Web UI (8080)
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run both API and Web UI
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

CMD ["/docker-entrypoint.sh"]
