const mongoose = require('mongoose');
mongoose.Promise = global.Promise;

const db = {};

db.mongoose = mongoose;
db.request = require("./request.model");

db.ROLES = ["user", "admin", "guest"];

module.exports = db;