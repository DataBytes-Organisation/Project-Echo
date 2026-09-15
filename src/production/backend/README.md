# Project Echo Backend

The Project Echo Backend is a FastAPI service responsible for receiving detection data, interacting with MongoDB, and providing API endpoints for other Project Echo components such as the Engine and HMI.

## Backend Quick Start - T2 2026

This guide explains how to start and verify the Project Echo Backend independently using Docker Compose.

### Prerequisites

Before starting, install:

- Git
- Docker Desktop

Ensure Docker Desktop is running before executing the Docker commands below.

### 1. Update the Repository

From the Project Echo repository root:

```powershell
git switch main
git pull --ff-only origin main
```

For development work, create or switch to your own feature branch rather than making changes directly on `main`.

### 2. Navigate to the Docker Compose Directory

```powershell
cd .\src\deployment\docker
```

The main Backend-related Docker services are:

- `echo_store` - MongoDB
- `echo_api` - FastAPI Backend API
- `mongo-express` - MongoDB web administration interface

### 3. Start MongoDB

```powershell
docker compose up -d --build echo_store
```

Verify that MongoDB is running:

```powershell
docker compose ps echo_store
```

MongoDB is exposed locally on:

```text
localhost:27017
```

### 4. Start the Backend API

```powershell
docker compose up -d --build echo_api
```

Verify the Backend container:

```powershell
docker compose ps echo_api
```

Check the Backend logs:

```powershell
docker compose logs --tail 100 echo_api
```

A successful startup should include messages similar to:

```text
Uvicorn running on http://0.0.0.0:9000
Application startup complete.
```

### 5. Verify the Backend API

Check the API root:

```powershell
Invoke-WebRequest http://localhost:9000/ -UseBasicParsing |
Select-Object StatusCode, Content
```

A successful response should return HTTP status `200`.

Swagger documentation is available at:

```text
http://localhost:9000/docs
```

The Engine ingestion endpoint can be found in Swagger as:

```text
POST /engine/event
```

### 6. Test Engine Event Ingestion

The Engine sends detection events to:

```text
POST http://localhost:9000/engine/event
```

The current event payload contains:

- timestamp
- sensorId
- species
- microphoneLLA
- animalEstLLA
- animalTrueLLA
- animalLLAUncertainty
- audioClip
- confidence
- sampleRate

A successful request returns HTTP `201`.

### 7. Verify MongoDB Persistence

Open the MongoDB shell:

```powershell
docker exec -it ts-mongodb-cont mongo -u root --authenticationDatabase admin -p
```

Enter the locally configured MongoDB development password when prompted.

Select the Project Echo database:

```javascript
use EchoNet
```

View the latest stored event:

```javascript
db.events.find().sort({ _id: -1 }).limit(1).pretty()
```

Do not expose MongoDB passwords, API keys, tokens or other credentials in screenshots, documentation, commits or reports.

### 8. Stop the Backend Services

Stop the Backend API and MongoDB:

```powershell
docker compose stop echo_api echo_store
```

If Mongo Express is running:

```powershell
docker compose stop mongo-express
```

## Known Local Development Notes

Docker Compose may display a warning stating that the `version` attribute is obsolete. This warning currently does not prevent the Backend services from starting.

Mongo Express 1.0.2 may attempt to connect to the default hostname `mongo:27017` rather than the current MongoDB service/container hostname. This does not prevent the Backend API and MongoDB from operating independently and should be tracked as a configuration issue.

---

# TEAM PROJECT T3 11/2025 NOTES NGUYEN GIA KHANG TRIEU

# If you're interested in any of these features,use the link bellow.

# https://docs.google.com/document/d/1XtzbgMz1Yt6OSmrCM1hzhCpC7rgfgT1oK9T17tzbUnc/edit?usp=sharing

# 1. Update Engine to run 24/7 with continuous processing

# 2. WebSocket for real-time detections

# https://docs.google.com/document/d/1VxjDyqtyD9dx48-H1Za72js-g5GKcAFMDSzD3KdIQa0/edit?usp=sharing

# Backend API for Sensor Health