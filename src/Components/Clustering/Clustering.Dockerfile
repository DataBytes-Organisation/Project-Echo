# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    libtag1-dev \
    libtagc0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set a directory for the app
WORKDIR /usr/src/app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Command to run the script
CMD ["python", "./clustering.py"]