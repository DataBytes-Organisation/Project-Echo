# C13 Seed & Fixture Generator and B2.4 Analytics Fixture Validation

## Overview

This document describes the Sprint 2 Backend work completed for:

- **C13 - Backend Seed and Fixture Generator CLI**
- **B2.4 - Deterministic fixture support for Detection Insights and Analytics**

C13 provides repeatable, schema-valid Project Echo detection data for local development,
integration testing, analytics verification, and future Sprint 2 test scenarios.

B2.4 reuses the C13 fixtures to verify deterministic analytics totals against the existing
Project Echo insights implementation.

---

## C13 - Backend Seed and Fixture Generator CLI

### Location

tools/seed_detections.py

Run from:

src/production/backend

using the Project Echo Backend container.

### Purpose

The generator creates realistic Project Echo detection fixtures using the real
`DetectionCreate` schema.

Each generated detection includes:

* `timestamp`
* `sensorId`
* `species`
* `microphoneLLA`
* `animalEstLLA`
* `animalTrueLLA`
* `animalLLAUncertainty`
* `audioClip`
* `confidence`
* `sampleRate`
* `source_model`

Fixture payloads are validated with `DetectionCreate` before they are eligible for
database insertion.

### Deterministic generation

The CLI supports a configurable random seed.

Using the same:

* seed
* count
* date range
* species list
* sensor list
* run identifier

produces the same detection data.

Example:

docker exec -it ts-api-cont python -m tools.seed_detections `
    --count 12 `
    --start 2026-09-01 `
    --end 2026-09-12 `
    --species "Sus Scrofa" `
    --species "Dingo" `
    --species "Crimson Rosella" `
    --sensor-id 2 `
    --sensor-id 3 `
    --seed 374 `
    --run-id c13-thomas-demo-001 `
    --collection events

Expected deterministic distribution:

Total detections: 12

Species:
Sus Scrofa          4
Dingo               4
Crimson Rosella     4

Sensors:
2                    6
3                    6

---

## Supported CLI options

The generator supports:

```text
--count
--days
--start
--end
--species
--sensor-id
--seed
--run-id
--collection
--database
--environment
--dry-run
--cleanup
--confirm-cleanup
```

Use:

docker exec -it ts-api-cont python -m tools.seed_detections --help

for the complete current interface.


## Collection support

The current Project Echo Backend contains two related data paths:

events
detections

The generator therefore supports:

--collection events

for the current Engine/B1/B2 analytics flow, and:

--collection detections

for testing the REST detection service.

The default collection is:

events

because the current `/insights/overview` and `/insights/species` implementation reads
from the `events` collection.


## Fixture metadata

Generated records are marked with C13-specific metadata:

{
  "_fixture": {
    "generated": true,
    "generator": "c13_seed_detections",
    "runId": "example-run",
    "seed": 374,
    "environment": "local",
    "createdAt": "..."
  }
}

This metadata allows generated test data to be identified and safely removed without
targeting normal Project Echo records.

## Dry-run behaviour

Example:

docker exec -it ts-api-cont python -m tools.seed_detections `
    --count 20 `
    --species "Sus Scrofa" `
    --dry-run

Dry-run mode:

1. generates fixtures;
2. validates them with `DetectionCreate`;
3. prints a generation summary and sample fixture;
4. performs no database insertion.


## Cleanup safety

Cleanup is intentionally restricted.

A destructive cleanup requires:

--cleanup
--database
--environment
--confirm-cleanup

Production cleanup is rejected.

Cleanup only targets records where:

_fixture.generated = true
_fixture.generator = c13_seed_detections

A `--run-id` can be supplied to limit cleanup to one fixture run.

### Cleanup preview

docker exec -it ts-api-cont python -m tools.seed_detections `
    --cleanup `
    --database EchoNet `
    --environment local `
    --collection events `
    --run-id c13-thomas-demo-001 `
    --confirm-cleanup `
    --dry-run

Observed test result:

12 fixture(s) would be removed

The database still contained all 12 fixtures after this dry-run.

### Real cleanup

docker exec -it ts-api-cont python -m tools.seed_detections `
    --cleanup `
    --database EchoNet `
    --environment local `
    --collection events `
    --run-id c13-thomas-demo-001 `
    --confirm-cleanup

Observed result:

12 fixture(s) removed

A final MongoDB check confirmed:

0

matching records remained.


## C13 automated tests

Test file:

tests/test_seed_detections.py

Coverage includes:

* requested fixture count;
* `DetectionCreate` validation;
* deterministic output with the same seed;
* different output with a different seed;
* predictable species distribution;
* predictable sensor distribution;
* requested timestamp range;
* fixture metadata;
* safe cleanup filters;
* targeted cleanup by run ID;
* invalid date ranges;
* required database target;
* required explicit environment;
* cleanup confirmation;
* production cleanup rejection;
* dry-run database-write prevention.

Run:

docker exec -it ts-api-cont python -m pytest `
    -q tests/test_seed_detections.py

Observed result:

15 passed

# B2.4 - Analytics Fixture Validation

## Purpose

B2.4 reuses C13 deterministic fixtures to verify expected aggregate totals from the
existing Project Echo analytics implementation.

Test file:

tests/test_insights_fixtures.py

The B2.4 dataset contains:

12 detections
3 species
2 sensors

Expected species totals:

Sus Scrofa          4
Dingo               4
Crimson Rosella     4

Expected sensor totals:

Sensor 2            6
Sensor 3            6


## Analytics verified

The tests validate the existing analytics behaviour for:

/insights/overview
/insights/species

Expected overview values:

detections             = 12
uniqueSpecies          = 3
sensorsWithDetections  = 2

The species aggregation must return:

Crimson Rosella = 4
Dingo            = 4
Sus Scrofa       = 4

The sum of species counts must also equal the overview detection total:

4 + 4 + 4 = 12


## Database isolation

B2.4 automated tests do not write to the normal Project Echo collections.

Instead, the tests redirect the analytics module to dedicated collections:

_b2_4_test_events
_b2_4_test_microphones
_b2_4_test_nodes

inside the existing `EchoNet` database.

This approach was chosen because the current Project Echo MongoDB application user
does not have permission to create and drop arbitrary databases.

The dedicated test collections are cleaned before and after the tests.

The normal:

events
microphones
nodes

collections are not modified by the B2.4 automated test suite.

## Combined C13 + B2.4 regression test

Run:

docker exec -it ts-api-cont python -m pytest `
    -q tests/test_seed_detections.py `
    tests/test_insights_fixtures.py

Observed result:

22 passed

This confirms that:

* the C13 generator remains valid;
* deterministic fixtures are reusable by another Sprint 2 task;
* analytics return the expected known totals;
* cleanup and database safety behaviour continue to pass.

## Current limitations / dependencies

B2 Sprint 2 is intended to extend analytics with additional filters such as date,
species and sensor.

At the time this support work was completed:

* `/insights/overview` supports start/end filtering;
* `/insights/species` supports result limiting;
* the full Sprint 2 species/sensor filtering contract may depend on other B2 member work.

B2.4 should be extended with additional deterministic filter assertions once those
changes are merged.

This support task does not independently redesign or replace the B2 analytics API.

## Evidence summary

C13 evidence:

15 automated tests passed
12 real MongoDB fixtures inserted
4 / 4 / 4 deterministic species distribution verified
dry-run cleanup predicted 12 records
dry-run preserved all 12 records
real cleanup removed exactly 12 records
final matching record count = 0

B2.4 evidence:

C13 fixture generator reused
12 known detections
3 known species
2 known sensors
expected 4 / 4 / 4 aggregate verified
overview totals verified
analytics consistency verified
combined regression suite = 22 passed


