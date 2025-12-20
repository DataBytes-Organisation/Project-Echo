# Builder (Compilers and heavy lifting)
ARG BASE_IMAGE=python:3.10-slim-bullseye
FROM ${BASE_IMAGE} AS echo_engine_builder

WORKDIR /build

# Install build-time dependencies only
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
	libopenexr-dev \
	pkg-config \
	build-essential \
	&& rm -rf /var/lib/apt/lists/* 

# Create a virtual environment to isolate packages
RUN python -m venv /opt/venv 
# Enable venv for the following commands
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Optimization: Combine pip calls and use no-cache
RUN pip3 install --upgrade pip && \
	pip3 install --no-cache-dir -r requirements.txt 

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
	libopenexr25 \
	libgl1-mesa-glx \
	libglib2.0-0 \
	curl \
	gnupg \
	ca-certificates \
	&& echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
	&& curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
	&& apt-get update -y \
	&& apt-get install -y google-cloud-cli \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* 

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

CMD ["/app/echo_engine.sh"]
