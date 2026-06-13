FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and model artifacts
COPY src/ ./src/
COPY *.txt ./
COPY *.json ./

# The stream engine just needs python path set
ENV PYTHONPATH=/app/src

CMD ["python", "src/stream_engine.py"]