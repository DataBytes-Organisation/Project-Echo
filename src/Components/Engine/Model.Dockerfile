# Use an official TensorFlow serving image as the base image
FROM tensorflow/serving:2.3.0

RUN apt-get update && apt-get install -y build-essential

# Copy the model to the container
COPY models/generic_engine_pipeline_model/1 /models/generic_engine_pipeline_model/1

# Set the environment variable for the model name and version
ENV MODEL_NAME=generic_engine_pipeline_model
ENV MODEL_VERSION=1

# Expose the port for TensorFlow serving
EXPOSE 8501