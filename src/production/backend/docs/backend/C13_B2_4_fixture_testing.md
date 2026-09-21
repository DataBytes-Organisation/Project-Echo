# C13 Seed & Fixture Generator and B2.4 Analytics Fixture Validation

## Overview

This document describes the Sprint 2 Backend work completed for:

- **C13 - Backend Seed and Fixture Generator CLI**
- **B2.4 - Deterministic fixture support for Detection Insights and Analytics**

C13 provides repeatable, schema-valid Project Echo fixtures for local development,
integration testing, analytics verification, and downstream Sprint 2 tasks.

The generator supports both current Backend data paths:

- `events`
- `detections`

and generates collection-specific fixture structures appropriate to each path.

B2.4 reuses the C13 fixture generator to verify deterministic analytics totals and
filter behaviour against the existing Project Echo insights implementation.

---

# C13 - Backend Seed and Fixture Generator CLI

## Location

tools/seed_detections.py

Run from:

src/production/backend

using the Project Echo Backend container.

## Purpose

The generator creates deterministic Project Echo fixtures using the schema and
document structure appropriate to the selected collection.

For:

--collection events

the generator produces event-compatible documents for the current Engine/B1/B2
analytics flow.

For:

--collection detections

the generator produces detection documents validated against the existing
DetectionCreate schema.

This collection-specific behaviour ensures that generated fixtures match the data
contract expected by the selected Backend path.

## Collection-specific fixture generation

The current Project Echo Backend contains two related data paths:

events
detections

The generator therefore supports:

--collection events

for the current Engine/B1/B2 analytics flow, and:

--collection detections

for the REST detection service.

The default collection is:

events

because the current /insights/overview and /insights/species implementation reads
from the events collection.

Event fixtures include the fields required by the current event schema and analytics
pipeline, including the appropriate event source and structured location data.

Detection fixtures continue to use the DetectionCreate validation path.

## Deterministic generation

The CLI supports a configurable random seed.

Using the same:

seed
count
date range
species list
sensor list
run identifier
selected collection

produces deterministic fixture data.

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

Total fixtures: 12

Species:
Sus Scrofa          4
Dingo               4
Crimson Rosella     4

Sensors:
2                   6
3                   6

## Supported CLI options

The generator supports:

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

Use:

docker exec -it ts-api-cont python -m tools.seed_detections --help

for the complete current interface.

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

This metadata allows generated fixture data to be identified and safely removed
without targeting normal Project Echo records.

## Dry-run behaviour

Example:

docker exec -it ts-api-cont python -m tools.seed_detections `
    --count 20 `
    --species "Sus Scrofa" `
    --dry-run

Dry-run mode:

generates fixtures;
validates them against the collection-appropriate schema/structure;
prints the generation summary and sample fixture;
performs no database insertion.

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

A --run-id can be supplied to limit cleanup to one fixture run.

## Cleanup preview
docker exec -it ts-api-cont python -m tools.seed_detections `
    --cleanup `
    --database EchoNet `
    --environment local `
    --collection events `
    --run-id c13-thomas-demo-001 `
    --confirm-cleanup `
    --dry-run

Observed validation:

12 fixture(s) identified for cleanup
0 fixture(s) removed during dry-run

The fixture records remained present after the dry-run.

## Confirmed cleanup
docker exec -it ts-api-cont python -m tools.seed_detections `
    --cleanup `
    --database EchoNet `
    --environment local `
    --collection events `
    --run-id c13-thomas-demo-001 `
    --confirm-cleanup

Observed result:

12 fixture(s) removed

An isolated integration verification was also performed using a dedicated review
database:

Initial unrelated control records: 1
C13 fixtures inserted: 12
Total records after seeding: 13

Dry-run cleanup identified: 12
Dry-run cleanup removed: 0
Unrelated control record preserved: yes

Confirmed cleanup removed: 12
Final records remaining: 1

The final remaining record was the unrelated control record.

This confirms that targeted C13 cleanup removes only C13-tagged fixtures and leaves
unrelated records untouched.

## C13 automated tests

Test file:

tests/test_seed_detections.py

Coverage includes:

requested fixture count;
deterministic output using the same seed;
changed output using a different seed;
predictable species distribution;
predictable sensor distribution;
requested timestamp range;
fixture metadata;
safe cleanup filters;
targeted cleanup by run ID;
invalid date ranges;
required database target;
required explicit environment;
cleanup confirmation;
production cleanup rejection;
dry-run database-write prevention;
collection-specific event fixture generation;
collection-specific detection fixture generation;
event-schema compatibility;
filtered event read behaviour.

Run:

docker exec -it ts-api-cont python -m pytest `
    -q tests/test_seed_detections.py

Observed result:

19 passed

C13 was subsequently merged into main.

## B2.4 - Analytics Fixture Validation
### Purpose

B2.4 reuses the C13 deterministic fixture generator to verify expected aggregate totals
and filter behaviour from the existing Project Echo analytics implementation.

Test file:

tests/test_insights_fixtures.py

B2.4 uses the current C13 generator with:

collection_name="events"

so that the fixture structure matches the collection used by the analytics routes.

The deterministic B2.4 dataset contains:

12 events
3 species
2 sensors

Expected species totals:

Sus Scrofa          4
Dingo               4
Crimson Rosella     4

Expected sensor totals:

Sensor 2            6
Sensor 3            6

### Analytics verified

The B2.4 tests validate analytics behaviour for:

/insights/overview
/insights/species

Expected overview values:

detections             = 12
uniqueSpecies          = 3
sensorsWithDetections  = 2

Expected species aggregation:

Crimson Rosella = 4
Dingo           = 4
Sus Scrofa      = 4

The sum of the species totals must equal the overview detection total:

4 + 4 + 4 = 12

The deterministic suite also verifies:

species filtering;
sensor filtering;
date-range filtering;
combined species and sensor filtering.

Known expectations include:

species=Sus Scrofa          -> 4
sensorId=2                  -> 6
2026-09-03 to 2026-09-06    -> 4
species=Dingo + sensorId=3  -> 2

### Database isolation

B2.4 automated integration tests do not use the normal Project Echo analytics
collections directly.

Instead, the analytics module is redirected to dedicated test collections inside the
existing EchoNet database:

_b2_4_test_events
_b2_4_test_microphones
_b2_4_test_nodes

The following normal Project Echo collections are therefore not used as test targets:

events
microphones
nodes

The dedicated B2.4 test collections are cleared before and after the integration tests.

The cleanup uses document deletion rather than dropping the MongoDB collections
themselves, so an empty _b2_4_test_* collection may remain present after a test run.
The important isolation guarantee is that B2.4 fixture documents are removed and the
normal Project Echo collections are not modified.

### MongoDB integration test environment

The shared:

tests/conftest.py

patches:

pymongo.MongoClient

to:

mongomock.MongoClient

for the general tests/ package.

The current analytics aggregation pipeline uses MongoDB expressions including:

$type

inside timestamp normalisation.

The installed mongomock implementation does not support this aggregation expression
and raises:

OperationFailure: Unrecognized expression '$type'

This is a test-environment limitation rather than an analytics failure.

For this reason, the B2.4 analytics integration suite is run against the local Project
Echo MongoDB container using:

--noconftest

This bypasses the shared mongomock patch and allows the tests to exercise the real
MongoDB aggregation behaviour while still using the isolated _b2_4_test_*
collections.

### B2.4 analytics fixture tests

Run:

docker exec -it ts-api-cont python -m pytest `
    --noconftest `
    -q tests/test_insights_fixtures.py

Observed result:

11 passed, 1 warning

The warning is the existing Starlette python_multipart pending deprecation warning
and is unrelated to B2.4.

### Combined C13 + B2.4 regression test

Run:

docker exec -it ts-api-cont python -m pytest `
    --noconftest `
    -q tests/test_seed_detections.py `
       tests/test_insights_fixtures.py

Observed result:

30 passed, 1 warning

This consists of:

19 C13 generator tests
11 B2.4 analytics fixture tests

This confirms that:

the latest merged C13 generator remains valid;
B2.4 is compatible with the current collection-specific C13 interface;
deterministic fixtures are reusable by downstream Sprint 2 analytics work;
event fixtures use the appropriate collection-specific structure;
analytics return the expected known totals;
species, sensor, date-range, and combined filters return deterministic results;
cleanup and database safety behaviour continue to pass;
the analytics integration tests work against the real MongoDB aggregation pipeline;
normal Project Echo collections remain isolated from the B2.4 fixture dataset.

### Dependency status

B2.4 originally depended on:

C13 - Backend Seed and Fixture Generator
PR #1047

C13 has now been merged into main.

B2.4 has therefore been reconciled against the current C13 implementation.

The remaining B2.4 diff is limited to the intended analytics validation support and
documentation.

## Evidence summary

### C13 evidence
19 automated tests passed
collection-specific events and detections supported
12 deterministic fixtures generated for integration validation
4 / 4 / 4 deterministic species distribution verified
6 / 6 deterministic sensor distribution verified
dry-run cleanup identified exactly 12 C13 fixtures
dry-run removed no records
confirmed cleanup removed exactly 12 C13 fixtures
unrelated control record remained untouched
C13 merged into main
### B2.4 evidence
C13 fixture generator reused
collection_name="events" used for analytics fixtures
12 known deterministic events
3 known species
2 known sensors
4 / 4 / 4 species aggregation verified
6 / 6 sensor distribution verified
overview totals verified
species filtering verified
sensor filtering verified
date-range filtering verified
combined species + sensor filtering verified
isolated _b2_4_test_* MongoDB collections used
B2.4-only suite = 11 passed
combined C13 + B2.4 suite = 30 passed

The B2.4 integration suite is executed against the local MongoDB instance using
--noconftest because the shared mongomock test environment does not support the
$type aggregation expression used by the current analytics timestamp-normalisation
pipeline.