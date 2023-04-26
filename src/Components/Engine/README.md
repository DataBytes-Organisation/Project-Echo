# Engine

## Generic Engine Pipeline

This is an entry point and sample code for how to build an end to end pipeline from loading raw audio data and generating a model

## Optimised Engine Pipeline

This is a variant of the generic engine pipeline that execute a parallel pipeline - this is ready for producing the production model.

## Docker

Run these steps to execute the engine in docker

### Setup volumes

docker volume create myvolume

### Setup network

docker network create --driver bridge echo-net

### Model Server
(1) docker build --file Model.Dockerfile -t ts-echo-model .
(2) docker run -p 8501:8501 --name ts-echo-model-cont --network echo-net -d ts-echo-model

### Echo Engine
(1) docker build --file Engine.Dockerfile -t ts-echo-engine .
(2) docker run --name ts-echo-engine-cont -it --rm -v myvolume:/root --network echo-net ts-echo-engine 
