const mongoose = require("mongoose");

// Define the schema for the "Guest" collection
const guestSchema = new mongoose.Schema({
    userId: String,
    username: String,
    email: String,
    password: String,
    expiresAt: { type: Date, default: Date.now, expires: 1800 }, // TTL of 30 minutes (1800 seconds)
    roles: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Role"
      }
    ]
  });
// Create the "Guest" model
const Guest = mongoose.model('Guest', guestSchema);

// Export the model so that it can be used in other files
module.exports = Guest;