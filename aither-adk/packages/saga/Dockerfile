FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Install aither-adk from parent
COPY pyproject.toml setup.cfg* /adk/
COPY adk/ /adk/adk/
RUN pip install --no-cache-dir /adk

# Copy saga package
COPY packages/saga/ /app/packages/saga/

# Install saga deps
RUN pip install --no-cache-dir \
    pydantic>=2.0 \
    uvicorn>=0.27.0 \
    fastapi>=0.109.0 \
    httpx>=0.27.0 \
    pyyaml>=6.0

WORKDIR /app/packages/saga

# Prometheus URL for simulation integration (set to actual Prometheus service when available)
ENV SAGA_PROMETHEUS_URL=http://aitheros-prometheus:8178
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "saga_server.py", "--port", "8080"]
