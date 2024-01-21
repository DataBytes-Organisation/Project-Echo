# Dockerfile

# Use an official Node.js runtime as the base image
FROM node:latest

# Set the working directory in the Docker container
WORKDIR /usr/src/app

# Copy the entire contents of the current directory to the Docker image
COPY . .

# Change to the UI directory where your package.json file is located
WORKDIR /usr/src/app/ui

# Install your application's dependencies
RUN npm install

# The application runs on port 8080, so let Docker know about this
EXPOSE 8080

# The command to run your application
CMD [ "node", "server.js" ]
