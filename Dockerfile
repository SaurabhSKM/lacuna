FROM python:3.12-slim

# Install dependencies first, separately from the app code, so Docker's
# layer cache skips this (slow) step on rebuilds that only touch src/.
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
# Sample documents, so `build_index.py` can be run inside the container
# to get a working demo index without mounting anything.
COPY data/sample_docs/ ./data/sample_docs/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# fastembed caches ONNX model weights here. Pointing it inside the image
# keeps the download out of the container's writable layer at runtime,
# and lets the next line bake it in rather than fetching ~80MB on the
# first request of every new container.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# Run as a non-root user. The vector store directory is created and
# chowned here because it's a mount point at runtime and the process
# needs to be able to write to it.
RUN useradd --create-home --uid 10001 lacuna \
    && mkdir -p /app/chroma_db /app/data \
    && chown -R lacuna:lacuna /app /opt/fastembed
USER lacuna

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
