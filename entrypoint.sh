#!/bin/bash
set -e

echo "=== Running data ingestion (idempotent) ==="
python -m src.data.ingest_db

echo "=== Starting API server ==="
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
