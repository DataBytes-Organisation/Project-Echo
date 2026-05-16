# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y build-essential gcc bash

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full app into the container
COPY . .

# Expose port for the app
EXPOSE 9000

# Command to run the app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
