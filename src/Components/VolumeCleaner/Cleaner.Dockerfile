# Use the official Python image from the Docker Hub
FROM python:3.9-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container
COPY volumeCleanup.py /app/

# Command to run the Python script
CMD ["python", "volumeCleanup.py"]