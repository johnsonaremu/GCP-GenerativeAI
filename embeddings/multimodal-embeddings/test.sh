#! /bin/bash

curl -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d @request.json \
    "https://us-central1-aiplatform.googleapis.com/v1/projects/cloud-llm-preview2/locations/us-central1/publishers/google/models/multimodalembedding@001:predict"
