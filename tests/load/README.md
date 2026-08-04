# Project Echo End-to-End Load Testing

## Overview

This folder contains Artillery load tests created for Task D1.1: Reliability and Scale.

The current tests measure the performance of the Project Echo API retrieval route:

`GET /iot/nodes`

The tests are designed to establish a baseline and then apply a higher stress workload.

## Requirements

Before running the tests:

- Docker Desktop must be running.
- The Project Echo containers must be started.
- The API must be available at `http://localhost:9000`.
- Node.js and npm must be installed.

Artillery is executed using `npx`, so a global installation is not required.

## Test files

### `retrieval-baseline.yml`

Runs three phases:

- 2 users/second for 20 seconds
- 5 users/second for 30 seconds
- 10 users/second for 30 seconds

This test establishes the normal performance baseline.

### `retrieval-stress.yml`

Runs four phases:

- 5 users/second for 20 seconds
- 20 users/second for 30 seconds
- 50 users/second for 30 seconds
- 100 users/second for 20 seconds

This test checks how the API behaves under increasing traffic and a short spike.

## Running the tests

From the repository root:

```bash
./tests/load/run-load-tests.sh
