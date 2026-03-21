# Builder (Compilers and heavy lifting)
ARG BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS echo_engine_builder

WORKDIR /build

# Install build-time dependencies only
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3 \
	python3-pip \
	python3-venv \
	python3-dev \
	libopenexr-dev \
	pkg-config \
	build-essential \
	dos2unix \
	&& rm -rf /var/lib/apt/lists/* 

# Create a virtual environment to isolate packages
RUN python3 -m venv /opt/venv 
# Enable venv for the following commands
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Optimization: Combine pip calls and use no-cache
RUN pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt 

# Handle script formatting here so it doesn't create layers in the final image
COPY ./echo_engine.sh .
RUN apt-get update && apt-get install -y dos2unix && \
	dos2unix ./echo_engine.sh && \
	chmod +x ./echo_engine.sh 

# Runner (The final, slim image)
FROM ${BASE_IMAGE}

WORKDIR /app

# Install ONLY runtime libraries (not the -dev versions)
# We also add the gcloud CLI here in a single consolidated step
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	python3 \
	python3-distutils \
	libsndfile1 \
	libopenexr25 \
	libgl1-mesa-glx \
	libglib2.0-0 \
	curl \
	gnupg \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/* 

# Google Cloud CLI
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
	&& curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
	&& apt-get update -y \
	&& apt-get install -y google-cloud-cli \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* 

RUN ln -s /usr/bin/python3 /usr/bin/python 

COPY --from=echo_engine_builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=echo_engine_builder /build/echo_engine.sh ./

COPY yamnet_dir/ ./yamnet_dir/
COPY ./echo_engine.py ./
COPY ./echo_engine.json ./
COPY ./echo_credentials.json ./
COPY ./helpers ./helpers

# Setup GCloud config dir
RUN mkdir -p /root/.config/gcloud/

CMD ["/bin/bash", "/app/echo_engine.sh"]
