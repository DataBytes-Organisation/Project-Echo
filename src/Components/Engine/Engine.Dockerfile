# Use an official TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt ./

# Install any needed Python packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the application code into the container
COPY ../echo_engine.py .

# Expose the required port for your application (if needed)
# Replace 'PORT_NUMBER' with the actual port number your application uses
# EXPOSE PORT_NUMBER

# Define the default command to run when the container starts
CMD ["python", "echo_engine.py"]
