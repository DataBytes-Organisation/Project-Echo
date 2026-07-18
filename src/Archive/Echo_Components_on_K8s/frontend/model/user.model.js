const mongoose = require("mongoose");

const User = mongoose.model(
  "User",
  new mongoose.Schema({
    // _id: mongoose.Types.ObjectId,
    userId: String,
    username: String,
    email: String,
    password: String,
    roles: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Role"
      }
    ],
    gender: String,
    DoB: Date,
    address: [
      {
        country: String,
        state: String,
        homeaddress: String
      }
    ],
    organization: String,
    phonenumber: String,
  })
);

module.exports = User;