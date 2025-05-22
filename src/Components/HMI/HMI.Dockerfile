# Dockerfile

# Use an official Node.js runtime as the base image
FROM node:latest

# Set the working directory in the Docker container
WORKDIR /usr/src/app

# Copy the entire contents of the current directory to the Docker image
COPY . .

# Install express in /usr/src/app (for server.js)
RUN npm install express

# Install dependencies from /usr/src/app/ui (if package.json exists there)
WORKDIR /usr/src/app/ui
RUN npm install

# Install nodemon globally
RUN npm install -g nodemon

# Change back to /usr/src/app to run server.js
WORKDIR /usr/src/app

# The application runs on port 8080, so let Docker know about this
EXPOSE 8080

# The command to run your application
CMD [ "nodemon", "server.js" ]