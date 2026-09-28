const notificationFeed = require('./notifications');

const NOTIFICATION_STATE_COLLECTION = 'notificationState';

function createNotificationStore({ donationClient, dbState }) {
  async function getEchoNetDb() {
    if (!dbState.connectedDB) {
      await donationClient.connect();
      dbState.connectedDB = donationClient.db('EchoNet');
      console.log('Reconnected to MongoDB for notifications');
    }
    return dbState.connectedDB;
  }

  async function listNotifications() {
    const db = await getEchoNetDb();

    const donations = await db.collection('donations').find({}).toArray();
    const users = await donationClient
      .db('UserSample')
      .collection('users')
      .find({}, { projection: { email: 1, username: 1, createdAt: 1 } })
      .toArray();
    const stateRows = await db.collection(NOTIFICATION_STATE_COLLECTION).find({}).toArray();

    return notificationFeed.applyState(
      notificationFeed.buildFeed({ donations, users }),
      stateRows
    );
  }

  async function setNotificationFlags(ids, flags) {
    if (ids.length === 0) return 0;

    const db = await getEchoNetDb();
    const operations = ids.map(id => ({
      updateOne: {
        filter: { _id: id },
        update: { $set: { ...flags, updatedAt: new Date() } },
        upsert: true
      }
    }));

    const result = await db.collection(NOTIFICATION_STATE_COLLECTION).bulkWrite(operations);
    return result.upsertedCount + result.modifiedCount;
  }

  async function close() {}

  return { listNotifications, setNotificationFlags, close };
}

module.exports = { createNotificationStore };
