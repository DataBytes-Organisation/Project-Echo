# Builder (Compilers and heavy lifting)
# bullseye (Debian 11) reached full end-of-life 2026-08-31; its apt archive
# is being frozen/migrated, which breaks package installs unpredictably.
# bookworm (Debian 12) is the current supported release.
ARG BASE_IMAGE=python:3.10-slim-bookworm
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
	libopenexr-3-1-30 \
	libgl1-mesa-glx \
	libglib2.0-0 \
	curl \
	ca-certificates \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

# Google Cloud CLI, installed from Google's own tarball rather than through
# apt/apt-key + Google's apt repo. This avoids pulling in gnupg/gnupg2 as an
# apt dependency purely to verify the repo signature - the previous approach
# broke when Debian's bullseye-security archive started rejecting that
# dependency chain around bullseye's 2026-08-31 end-of-life.
RUN curl -sSL -o /tmp/gcloud.tar.gz \
		https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz \
	&& tar -xzf /tmp/gcloud.tar.gz -C /usr/local \
	&& rm /tmp/gcloud.tar.gz \
	&& /usr/local/google-cloud-sdk/install.sh --quiet --path-update false --usage-reporting false
ENV PATH="/usr/local/google-cloud-sdk/bin:$PATH"

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
