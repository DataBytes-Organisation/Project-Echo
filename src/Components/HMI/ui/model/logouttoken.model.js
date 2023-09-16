const mongoose = require("mongoose");

const logouttoken = mongoose.model(
  "logouttoken",
  new mongoose.Schema({
    jwtToken : String
  })
);

module.exports = logouttoken;
