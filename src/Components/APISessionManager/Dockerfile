FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm install
RUN npm install dotenv

COPY . .

EXPOSE 3123

CMD ["node", "index.js"]