FROM us-central1-docker.pkg.dev/videosearch-comparison/videosearch/webapp-base:v1

WORKDIR /app

# Fix missing cffi backend + add Cloud Run API client for job triggering
RUN python -m pip install --no-cache-dir --force-reinstall cffi cryptography \
    && python -m pip install --no-cache-dir google-cloud-run

COPY app/ app/
COPY static/ static/
COPY ingest.py .
COPY ingest_new.py .
COPY reingest_pipeline.py .

ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
