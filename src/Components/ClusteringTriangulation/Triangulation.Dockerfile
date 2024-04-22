FROM python:3.8-slim

# Set a directory for the app
WORKDIR /usr/src/app

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Command to run the script
CMD ["python", "./TriangulationClustering.py"]