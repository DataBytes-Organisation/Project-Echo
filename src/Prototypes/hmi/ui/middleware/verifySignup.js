const db = require("../model");
const ROLES = db.ROLES;
const User = db.user;

checkDuplicateUsernameOrEmail = (req, res, next) => {
  // Username
  User.findOne({
    username: req.body.username
  }).exec((err, user) => {
    if (err) {
      res.status(500).send({ message: err });
      return {status: "failed", message: "Find Username Error: " + err};
    }

    if (user) {
      res.status(400).send({ message: "Failed! Username is already in use!" });
      return {status: "failed", message: "Find Username Error: Username already in use"};
    }

    // Email
    User.findOne({
      email: req.body.email
    }).exec((err, user) => {
      if (err) {
        res.status(500).send({ message: err });
        return {status: "failed", message: "Find Email Error: " + err};
      }

      if (user) {
        res.status(400).send({ message: "Failed! Email is already in use!" });
        return {status: "failed", message: "Find Email Error: Email already in use"};
      }

      next();
    });
  });
};

checkRolesExisted = (req, res, next) => {
  if (req.body.roles) {
    for (let i = 0; i < req.body.roles.length; i++) {
      if (!ROLES.includes(req.body.roles[i])) {
        res.status(400).send({
          message: `Failed! Role ${req.body.roles[i]} does not exist!`
        });
        return {status: "failed", message: "Find Role Error: Role does not exist"};
      }
    }
  }

  next();
};

const verifySignUp = {
  checkDuplicateUsernameOrEmail,
  checkRolesExisted
};

module.exports = verifySignUp;