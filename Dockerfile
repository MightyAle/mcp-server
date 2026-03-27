FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY embed_service.py .
COPY mcp_server.py .

EXPOSE 8000

CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]

