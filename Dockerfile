FROM python:3.13-slim

# System libs needed by opencv + mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 \
        libgles2 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY sport_companion_ai/ ./sport_companion_ai/
COPY api/ ./api/
COPY examples/ ./examples/

RUN pip install --no-cache-dir -e ".[api]"

EXPOSE 8000
# Default: HTTP API. CLI demo still callable with an explicit override:
#   docker run --rm -v "$PWD":/data <image> \
#     python examples/analyze_squat.py /data/squat.mp4 --exercise squat
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
