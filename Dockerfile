FROM python:3.11-slim

WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY artifacts/ ./artifacts/
COPY sql/ ./sql/

# Copy and prepare entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set defaults
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
