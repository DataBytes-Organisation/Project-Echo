FROM python:3.9-slim-bullseye

USER root
WORKDIR /app

# Install System Dependencies.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libopenexr-dev \
    dos2unix \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install Python Packages Safely
COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0" "tensorflow==2.15.0" -r requirements.txt

# 4. Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
	&& curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
	&& apt-get update -y \
	&& apt-get install -y google-cloud-cli \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* 

# Setup GCloud config dir
RUN mkdir -p /root/.config/gcloud/

# 5. Copy Application Code AND Credentials
WORKDIR /app
COPY ./light_echo_engine.py ./
COPY ./light_echo_engine.json ./
COPY ./light_echo_credentials.json ./
COPY ./helpers ./helpers

# 6. Run the Engine
CMD ["python", "light_echo_engine.py"]