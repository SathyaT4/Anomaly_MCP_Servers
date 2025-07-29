# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system-level dependencies (needed for ML & plotting)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libatlas-base-dev gfortran libpng-dev libfreetype6-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Copy your application files into the container
COPY anamoly/ /app/

# Install Python dependencies (FastAPI, scikit-learn, matplotlib, uvicorn)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi regex pandas python-multipart uvicorn scikit-learn matplotlib

# Expose FastAPI's default port
EXPOSE 3000

# Start FastAPI server when the container runs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]

