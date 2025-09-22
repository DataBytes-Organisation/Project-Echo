FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc \
	&& rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
	&& apt-get purge -y build-essential gcc && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
COPY . .
RUN rm -f app/echo_config.json .env || true
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 9000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","9000"]
