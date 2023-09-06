const mongoose = require("mongoose");

const Request = mongoose.model(
  "Request",
  new mongoose.Schema({
    // _id: mongoose.Types.ObjectId,
    requestId: String,
    username: String,
    animal: String,
    requestingToChange: String,
    initial: String,
    modified: String,
    source: String,
    date: Date,
    status: String,
   
  })
);

module.exports = Request;
