# Use the official Node.js image as a base
FROM node:14

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install required dependencies
RUN npm install

# Copy the rest of the application code into the container
COPY . .

# Expose the port the application will run on
EXPOSE 8080

# Start the application
CMD [ "node", "server.js" ]
