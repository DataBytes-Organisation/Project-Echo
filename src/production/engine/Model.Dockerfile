    # Use an official TensorFlow serving image as the base image
    # FROM emacski/tensorflow-serving
    ARG BASE_IMAGE
    FROM ${BASE_IMAGE}
    # RUN apt-get update && apt-get install -y build-essential

    # Copy the model to the container
    COPY models/ ./models/
    COPY models.config ./models/


    # make the container directory for credentials
    # RUN mkdir -p ~/.config/gcloud/

    # Set the environment variable for the model name and version
    # ENV MODEL_NAME=echo_model
    # ENV MODEL_VERSION=1

    # Expose the port for TensorFlow serving
    EXPOSE 8501