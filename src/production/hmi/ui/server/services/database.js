const { MongoClient } = require("mongodb");

async function connectDatabases(config) {
  const donationClient = new MongoClient(config.mongodbUri, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });
  await donationClient.connect();
  const echoNetDb = donationClient.db("EchoNet");

  return {
    donationClient,
    echoNetDb,
    close() {
      return donationClient.close();
    },
  };
}

module.exports = { connectDatabases };
