# Use an official Python runtime as a parent image
FROM python:3.11-alpine

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential gcc

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full app into the container
COPY . .

# Expose port for the app
EXPOSE 8080

# Command to run the app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
