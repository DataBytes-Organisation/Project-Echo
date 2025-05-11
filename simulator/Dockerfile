FROM python:3.9-slim

# Install ffmpeg for audio playback
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY local_audio/ ./local_audio/
COPY simulate_audio.py .

CMD ["python", "simulate_audio.py"]
