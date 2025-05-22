const mongoose = require("mongoose");

const User = mongoose.model("User",
  new mongoose.Schema({
    username: String,
    email: String,
    password: String,
    mfaEnabled: { type: Boolean, default: false },
    roles: [{
      type: mongoose.Schema.Types.ObjectId,
      ref: "Role",
    }],
    status: { type: String, default: "Active" },
    gender: String,
    DoB: String,
    address: mongoose.Schema.Types.Mixed, // for array or object
    organization: String,
    notificationAnimals: Array,
    emailNotifications: Array,
    phonenumber: String,
    userId: String,
    
    // Add visit tracking fields
    visitCount: { type: Number, default: 0 },
    lastVisit: { type: Date }
  }, { collection: "users" }) // explicitly target collection
);

module.exports = User;