FROM python:3.9

WORKDIR /app

COPY requirements.txt .
COPY echo_credentials.json .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["python", "system_manager.py"]
