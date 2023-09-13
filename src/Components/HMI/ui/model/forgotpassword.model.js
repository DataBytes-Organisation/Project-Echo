const mongoose = require("mongoose");

const forgotpassword = mongoose.model(
  "forgotpassword",
  new mongoose.Schema({
    userId: String,
    newPassword: String,
    modified_date: Date
  })
);

module.exports = forgotpassword;
