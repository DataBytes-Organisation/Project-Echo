db = db.getSiblingDB("EchoNet");

// NOTE: Credentials removed from source control. Supply via environment / secret during container start.
// Example (do not commit real values):
//   MONGO_APP_USER, MONGO_APP_PASS
db.createUser({
  user: _getEnv("MONGO_APP_USER", "replace_me"),
  pwd: _getEnv("MONGO_APP_PASS", "replace_me"),
  roles: [ { role: "readWrite", db: "EchoNet" } ],
});

function _getEnv(name, fallback){
  try { return cat(`/run/secrets/${name}`).trim(); } catch(e) {}
  return fallback;
}

db.createCollection("events");
db.createCollection("microphones");
db.createCollection("movements");
db.createCollection("species");

//const eventsData = JSON.parse(cat('/docker-entrypoint-initdb.d/events.json'));
//db.events.insertMany(eventsData);

const microphonesData = JSON.parse(cat('/docker-entrypoint-initdb.d/microphones.json'));
db.microphones.insertMany(microphonesData);

const movementsData = JSON.parse(cat('/docker-entrypoint-initdb.d/movements.json'));
db.movements.insertMany(movementsData);

const speciesData = JSON.parse(cat('/docker-entrypoint-initdb.d/species.json'));
db.species.insertMany(speciesData);



