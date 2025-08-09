FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
ENV MAX_CONCURRENCY=2
EXPOSE 8000

# 生产建议加 --workers >1；此处先单进程，便于调试
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
