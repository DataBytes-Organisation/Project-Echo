// C7.1 - Create indexes on detections collection
db = db.getSiblingDB("EchoNet");

db.detections.createIndex(
  { timestamp: -1, species: 1 },
  { name: "idx_timestamp_species" }
);

print("Index idx_timestamp_species created successfully.");
