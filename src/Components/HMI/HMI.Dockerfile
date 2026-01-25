FROM node:22-alpine

ENV NODE_ENV=production
WORKDIR /usr/src/app/ui

# Layer Caching: Copy ONLY package files first
# This prevents re-installing node_modules.
COPY ui/package*.json ./

# Install All Dependencies (dev and production)
# Use npm install for flexibility with all dependencies
RUN npm install && \
    npm cache clean --force

COPY ui/ .

EXPOSE 8080

CMD [ "node", "server.js" ]
