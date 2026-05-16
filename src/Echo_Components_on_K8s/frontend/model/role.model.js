const mongoose = require("mongoose");

const Role = mongoose.model(
  "Role",
  new mongoose.Schema({
    // _id: mongoose.Types.ObjectId,
    name: String
  })
);

module.exports = Role;