# Use an official TensorFlow GPU image as the base image
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
#FROM python:3.9
USER root
# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt ./

# Copy yamnet to the container
COPY yamnet_dir/ ./yamnet_dir/

# Install any needed Python packages specified in requirements.txt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y libopenexr-dev dos2unix
RUN apt-get install pkg-config
RUN python3 -m pip install --upgrade pip
RUN pip download -r requirements.txt
RUN pip install -r requirements.txt
#RUN pip install tensorflow==2.15.0
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y

# make the container directory for credentials
WORKDIR /root
RUN mkdir -p .config/gcloud/

# Copy the rest of the application code into the container
WORKDIR /app
COPY ./echo_engine.py ./
COPY ./echo_engine.sh ./
COPY ./echo_engine.json ./
COPY ./echo_credentials.json ./

RUN chmod +x ./echo_engine.sh
RUN dos2unix ./echo_engine.sh

CMD ["/app/echo_engine.sh"]
