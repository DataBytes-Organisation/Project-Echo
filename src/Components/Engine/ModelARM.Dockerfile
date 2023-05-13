# Use an official TensorFlow serving image as the base image
FROM emacski/tensorflow-serving:2.5.1

# Copy the model to the container
COPY models/ ./models/

# Set the environment variable for the model name and version
ENV MODEL_NAME=echo_model
ENV MODEL_VERSION=1

# Expose the port for TensorFlow serving
EXPOSE 8501