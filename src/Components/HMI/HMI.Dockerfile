FROM node:22-alpine

ENV NODE_ENV=production
WORKDIR /usr/src/app/ui

# Layer Caching: Copy ONLY package files first
# This prevents re-installing node_modules.
COPY ui/package*.json ./

# Install Production Dependencies Only
# 'npm ci' is faster and more reliable than 'install'.
# '--omit=dev' skips devDependencies (saves huge space).
# 'npm cache clean' removes the installation cache.
RUN npm ci --omit=dev && \
    npm cache clean --force

COPY ui/ .

EXPOSE 8080

CMD [ "node", "server.js" ]
