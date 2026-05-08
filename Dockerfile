FROM python:3.13-slim

# System libs needed by opencv + mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY sport_companion_ai/ ./sport_companion_ai/
COPY examples/ ./examples/

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "examples/analyze_squat.py"]
CMD ["--help"]
