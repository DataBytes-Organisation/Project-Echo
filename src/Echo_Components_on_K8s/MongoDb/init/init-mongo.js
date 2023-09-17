db = db.getSiblingDB("EchoNet");

db.createUser({
  user: "modelUser",
  pwd: "EchoNetAccess2023",
  roles: [
    {
      role: "readWrite",
      db: "EchoNet",
    },
  ],
});

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



