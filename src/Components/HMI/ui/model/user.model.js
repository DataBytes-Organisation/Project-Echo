const mongoose = require("mongoose");

const User = mongoose.model(
  "User",
  new mongoose.Schema({
    _id: mongoose.Types.ObjectId,
    username: String,
    email: String,
    password: String,
    roles: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Role"
      }
    ]
  })
);

module.exports = User;