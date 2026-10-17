# Optional containerised ml-service (P8.1/P8.2): CPU-only torch. Models come
# from ./models (mounted) or MODELS_URL + MODELS_SHA256 (fetched at start).
FROM python:3.11-slim
WORKDIR /app/ml-service
ENV PIP_NO_CACHE_DIR=1 TORCH_DEVICE=cpu MODEL_FP16=false HF_HOME=/app/.hf
COPY ml-service/pyproject.toml ./
COPY ml-service/ ./
COPY config/ /app/config/
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install ".[models]"
EXPOSE 8001
CMD ["sh", "-c", "python -m app.artifacts fetch && uvicorn app.main:app --host 0.0.0.0 --port 8001"]
