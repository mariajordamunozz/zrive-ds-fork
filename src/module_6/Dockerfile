FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps required by some wheels (pyarrow/numpy fall back to source on rare platforms).
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first to leverage Docker layer cache.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy source, model and data. Symlinks are dereferenced into the image.
COPY src/ ./src/
COPY data/ ./data/
COPY bin/ ./bin/

EXPOSE 8000

CMD ["python", "-m", "src.app"]
