#!/bin/bash

FILE=/root/.config/gcloud/application_default_credentials.json
if [ -f "$FILE" ]; then
    echo "$FILE exists so skipping gcloud authentication"
else 
    echo "$FILE does not exist."
    gcloud auth application-default login
fi

python echo_engine.py
