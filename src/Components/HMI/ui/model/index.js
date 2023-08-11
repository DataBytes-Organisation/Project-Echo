const mongoose = require('mongoose');
mongoose.Promise = global.Promise;

const db = {};

db.mongoose = mongoose;

db.user = require("./user.model");
db.role = require("./role.model");
db.guest = require("./guest.model")
db.request = require("./request.model");

db.ROLES = ["user", "admin", "guest"];

module.exports = db;