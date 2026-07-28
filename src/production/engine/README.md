# Engine

## Generic Engine Pipeline

This is an entry point and sample code for how to build an end to end pipeline from loading raw audio data and generating a model

## Optimised Engine Pipeline

This is a variant of the generic engine pipeline that execute a parallel pipeline - this is ready for producing the production model.

## Docker

***NOTE Before running this, please ensure you have trained a model and placed it in models/echo_model/1/***

Run these steps to execute the engine in docker

### Setup volumes

```
docker volume create myvolume
```

### Setup network

```
docker network create --driver bridge echo-net
```

### Model Server

```
docker build --file Model.Dockerfile -t ts-echo-model .
```

```
docker run -p 8501:8501 --name ts-echo-model-cont --network echo-net -d ts-echo-model
```

### Echo Engine

```
docker build --file Engine.Dockerfile -t ts-echo-engine .
```

```
docker run --name ts-echo-engine-cont -it --rm -v myvolume:/root --network echo-net ts-echo-engine 
```