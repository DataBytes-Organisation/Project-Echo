# Engine

## Docker

### Model Server
(1) docker build --file Model.Dockerfile -t ts-echo-model .
(2) docker run -p 8501:8501 --name ts-echo-model-cont -t ts-echo-model

### Echo Engine
(1) docker build --file Engine.Dockerfile -t ts-echo-engine .
(2) docker run --name ts-echo-engine-cont -t ts-echo-engine

