const mongoose = require("mongoose");

const forgotpassword = mongoose.model(
  "forgotpassword",
  new mongoose.Schema({
    userId: String,
    newPassword: String,
    modifiedDate: Date
  })
);

module.exports = forgotpassword;
