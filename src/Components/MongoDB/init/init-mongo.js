// ----------------------------------------------
// INITIALIZE API DATA
// ----------------------------------------------

apidb = db.getSiblingDB("EchoNet");

apidb.createUser({
  user: "modelUser",
  pwd: "EchoNetAccess2023",
  roles: [
    {
      role: "readWrite",
      db: "EchoNet",
    },
  ],
});

apidb.createCollection("events");
apidb.createCollection("microphones");
apidb.createCollection("movements");
apidb.createCollection("species");

//const eventsData = JSON.parse(cat('/docker-entrypoint-initdb.d/events.json'));
//db.events.insertMany(eventsData);

const microphonesData = JSON.parse(cat('/docker-entrypoint-initdb.d/microphones.json'));
apidb.microphones.insertMany(microphonesData);

const movementsData = JSON.parse(cat('/docker-entrypoint-initdb.d/movements.json'));
apidb.movements.insertMany(movementsData);

const speciesData = JSON.parse(cat('/docker-entrypoint-initdb.d/species.json'));
apidb.species.insertMany(speciesData);


// ----------------------------------------------
// INITIALIZE HMI SAMPLE DATA
// ----------------------------------------------
userdb = db.getSiblingDB("UserSample")
userdb.guests.ensureIndex({ email: 1 }, { unique: true, dropDups: true })
userdb.users.ensureIndex({ username:1, email: 1 }, { unique: true, dropDups: true })

const requests = JSON.parse(cat('/docker-entrypoint-initdb.d/request-seed.json'));
userdb.requests.insertMany(requests);

const roles = JSON.parse(cat('/docker-entrypoint-initdb.d/role-seed.json'));
userdb.roles.insertMany(roles);

const users = JSON.parse(cat('/docker-entrypoint-initdb.d/user-seed.json'));
userdb.users.insertMany(users);


userdb.guests.insertMany([
        { 
          "UserId": "HMITest1", 
          "username": "guest_tester1_0987654321", 
          "email": "guest1@echo.com", 
          "password": "$2a$08$bl0L01z7GxyFych1ZI1iXemfmAEGPEp.OQvJg2Sh3gnRO3QOGQZ4y",
          "roles": [
            {
              "_id": "64be1d0f05225843178d91d7"
            },
          ], 
          "__v": 0,
          "expiresAt": new Date(Date.now() + 30*60*1000) // 30 mins from initialization
        },

        {
          "userId": "HMITest1",
          "username": "guest_tester1_0987654321",
          "email": "guest2@echo.com",
          "password": "$2a$08$bl0L01z7GxyFych1ZI1iXemfmAEGPEp.OQvJg2Sh3gnRO3QOGQZ4y",
          "roles": [
            {
              "_id": "64be1d0f05225843178d91d7"
            }
          ],
          "expiresAt": new Date(Date.now() + 30*60*1000) // 30 mins from initialization
        }
    ])


