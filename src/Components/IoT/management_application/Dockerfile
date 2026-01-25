
FROM python:3.9-slim
WORKDIR /app
# Install system libraries for sounddevice, OpenCV, numpy, and scientific packages
RUN apt update && apt install -y \
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
	--no-install-recommends
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "client_pi.py"]
