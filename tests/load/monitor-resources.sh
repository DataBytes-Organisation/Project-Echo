#!/bin/bash

OUTPUT_FILE="${1:-tests/load/results/resources-during.csv}"

echo "timestamp,container,cpu_percent,memory_usage,memory_percent,net_io,block_io" > "$OUTPUT_FILE"

echo "Monitoring Docker resources..."
echo "Writing results to: $OUTPUT_FILE"
echo "Press Control+C to stop."

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    docker stats \
      --no-stream \
      --format "{{.Name}},{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}},{{.NetIO}},{{.BlockIO}}" \
      ts-api-cont ts-mongodb-cont echo-redis |
    while IFS= read -r line; do
        echo "$TIMESTAMP,$line" >> "$OUTPUT_FILE"
    done

    sleep 2
done
