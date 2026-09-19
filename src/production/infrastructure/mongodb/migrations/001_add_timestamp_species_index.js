// This file creates one index which is a shortcut for searching fast and easily
// on the detections collection, sorted by timestamp and species. 

const migrationDb = db.getSiblingDB("EchoNet");

migrationDb.detections.createIndex(
  { timestamp: -1, species: 1 },
  { name: "idx_timestamp_species" }
);

print("Done: index created on detections collection");
