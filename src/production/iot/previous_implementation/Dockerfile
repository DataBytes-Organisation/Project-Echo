
FROM python:3.9-slim-bookworm
WORKDIR /app
# Install system libraries for sounddevice, OpenCV, numpy, and scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
	portaudio19-dev \
	libgl1 \
	libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender1 \
	libsndfile1 \
	ffmpeg \
	libatlas3-base \
	liblapack-dev \
	gfortran \
	build-essential \
	gcc \
	&& rm -rf /var/lib/apt/lists/*
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "client_pi.py"]
