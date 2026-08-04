#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOAD_DIR="$ROOT_DIR/tests/load"
RESULTS_DIR="$LOAD_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "Checking Project Echo API..."
curl --fail --silent "http://localhost:9000/iot/nodes" >/dev/null

echo "Saving Docker resource usage before testing..."
docker stats --no-stream > "$RESULTS_DIR/resources-before.txt"

echo "Running retrieval baseline test..."
npx artillery@latest run \
  --output "$RESULTS_DIR/retrieval-baseline.json" \
  "$LOAD_DIR/retrieval-baseline.yml"

echo "Running retrieval stress test..."
npx artillery@latest run \
  --output "$RESULTS_DIR/retrieval-stress.json" \
  "$LOAD_DIR/retrieval-stress.yml"

echo "Saving Docker resource usage after testing..."
docker stats --no-stream > "$RESULTS_DIR/resources-after.txt"

echo "Load tests completed."
echo "Raw reports are available in: $RESULTS_DIR"
