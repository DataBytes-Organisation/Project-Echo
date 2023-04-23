# Use an official TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt ./

# Install any needed Python packages specified in requirements.txt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y libopenexr-dev
RUN python3 -m pip install --upgrade pip

RUN pip download -r requirements.txt

RUN pip install -r requirements.txt

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y

# Copy the rest of the application code into the container
COPY ./echo_engine.py ./
COPY ./echo_engine.sh ./
COPY ./echo_engine.json ./

# Expose the required port for your application (if needed)
# Replace 'PORT_NUMBER' with the actual port number your application uses
# EXPOSE PORT_NUMBER

# Define the default command to run when the container starts

# CMD ["bash"]

CMD ["/app/echo_engine.sh"]

# CMD ["python3", "echo_engine.py"]
