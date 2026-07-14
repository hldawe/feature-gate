FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Verify local model directory exists at build time
RUN python -c "import os; assert os.path.isdir('models/all-MiniLM-L6-v2'), \
    'models/all-MiniLM-L6-v2 not found — place the model directory in models/ before building'"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
