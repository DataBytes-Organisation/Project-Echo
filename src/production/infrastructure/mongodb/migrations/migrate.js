// This file checks: has migration 001 already run before?
// If not, run it. If yes, skip it.

const migrationsDb = db.getSiblingDB("EchoNet");
const historyBook = migrationsDb.migrations_history;

const alreadyDone = historyBook.findOne({ _id: "001" });

if (alreadyDone) {
  print("Already done, skipping migration 001");
} else {
  print("Running migration 001 now...");
  try {
    load("/migrations/001_add_timestamp_species_index.js");
    historyBook.insertOne({ _id: "001", appliedAt: new Date() });
    print("Migration 001 done and saved to history");
  } catch (err) {
    print("ERROR: Migration 001 failed - " + err.message);
    quit(1);
  }
}
