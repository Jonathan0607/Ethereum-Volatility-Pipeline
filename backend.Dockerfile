FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y g++ build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pybind11

COPY src/ ./src/
COPY research/ ./research/
COPY live_recon.py .
COPY scheduler.py .
COPY setup.py .
COPY *.txt ./
COPY *.json ./

RUN python setup.py build_ext --inplace
RUN mv *.so src/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
