# Kokao Engine Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY pyproject.toml .

# Install package
RUN pip install -e .

# Expose port for API
EXPOSE 8000

# Default command
CMD ["uvicorn", "kokao.kokao_hub.api:app", "--host", "0.0.0.0", "--port", "8000"]
