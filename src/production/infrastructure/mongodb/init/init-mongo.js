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
apidb.createCollection("nodes");

//const eventsData = JSON.parse(cat('/docker-entrypoint-initdb.d/events.json'));
//db.events.insertMany(eventsData);

const microphonesData = JSON.parse(cat('/docker-entrypoint-initdb.d/microphones.json'));
apidb.microphones.insertMany(microphonesData);

const movementsData = JSON.parse(cat('/docker-entrypoint-initdb.d/movements.json'));
apidb.movements.insertMany(movementsData);

const speciesData = JSON.parse(cat('/docker-entrypoint-initdb.d/species.json'));
apidb.species.insertMany(speciesData);

const nodesData = JSON.parse(cat('/docker-entrypoint-initdb.d/all-nodes-seed.json'));
apidb.nodes.insertMany(nodesData);

// ----------------------------------------------
// BULK UPDATE OPERATION
// ----------------------------------------------
const bulk = apidb.microphones.initializeOrderedBulkOp();
bulk.find({ status: "inactive" }).update({ $set: { status: "active" } });
bulk.find({ status: "deprecated" }).remove();
bulk.execute();

// ----------------------------------------------
// AGGREGATION PIPELINE OPERATION
// ----------------------------------------------
const aggregationResult = apidb.movements.aggregate([
  { $group: { _id: "$type", total: { $sum: 1 } } },
  { $sort: { total: -1 } }
]);

print("Movement Aggregation Result:", JSON.stringify(aggregationResult.toArray(), null, 2));


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

// Add notificationAnimals as an empty array for each user
users.forEach(user => {
  user.notificationAnimals = []; // Initialize notificationAnimals field
});

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


// ----------------------------------------------
// INITIALIZE DONATIONS COLLECTION
// ----------------------------------------------
const donations = [
  {
    amount: 3,
    status: 'succeeded',
    billing_details: { email: 'john@example.com' },
    created: 1743705600,
    type: 'One-Time'
  },
  {
    amount: 2,
    status: 'pending',
    billing_details: { email: 'jane@google.com' },
    created: 1743792000,
    type: 'Monthly'
  },
  {
    amount: 1,
    status: 'failed',
    billing_details: { email: 'hugo@outlook.com' },
    created: 1743619200,
    type: 'In-Kind'
  }
];


if (!apidb.getCollectionNames().includes("donations")) {
  apidb.createCollection("donations");
}

const dummyDonations = JSON.parse(cat('/docker-entrypoint-initdb.d/donations.json'));
const stripeDonations = JSON.parse(cat('/docker-entrypoint-initdb.d/donations-stripe-seed.json'));

// Combine both into one array
const combinedDonations = [...dummyDonations, ...stripeDonations];

// Insert into MongoDB
apidb.donations.insertMany(combinedDonations);
