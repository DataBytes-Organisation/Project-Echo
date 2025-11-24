ARG BASE_IMAGE=python:3.9-slim
FROM ${BASE_IMAGE} AS echo_simulator_builder

WORKDIR /build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	build-essential \
	&& rm -rf /var/lib/apt/lists/* 

RUN python -m venv /opt/venv 
ENV PATH="/opt/venv/bin:$PATH" 

COPY requirements.txt .

# Install dependencies into the virtual environment
RUN pip install --upgrade pip setuptools wheel && \
	pip install --no-cache-dir regex && \
	pip install --no-cache-dir -r requirements.txt 

# RUNNER (FINAL IMAGE)
# This is the actual image that will run in production
FROM ${BASE_IMAGE}

WORKDIR /app

# libgomp1 is needed for XGBoost, libsndfile1 for Librosa
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	libgomp1 \
	libsndfile1 \
	&& rm -rf /var/lib/apt/lists/*

# Copy the compiled environment from the builder stage
COPY --from=echo_simulator_builder /opt/venv /opt/venv

# Enable the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

COPY echo_credentials.json .
COPY src/ .

CMD ["python", "system_manager.py"]
