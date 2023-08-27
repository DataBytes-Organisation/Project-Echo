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
      return;
    }

    if (user) {
      // res.status(400).send({ message: "Failed! Username is already in use!" });
      res.status(400).send('<script>alert("Failed! Username is already in use!")</script>');
      return;
    }

    // Email
    User.findOne({
      email: req.body.email
    }).exec((err, user) => {
      if (err) {
        res.status(500).send({ message: err });
        return;
      }

      if (user) {
        // res.status(400).send({ message: "Failed! Email is already in use!" });
        res.status(400).send('<script>alert("Failed! Email is already in use!")</script>');
        return;
      }

      next();
    });
  });
};

checkRolesExisted = (req, res, next) => {
  if (req.body.roles) {
    for (let i = 0; i < req.body.roles.length; i++) {
      if (!ROLES.includes(req.body.roles[i])) {
        // res.status(400).send({
        //   message: `Failed! Role ${req.body.roles[i]} does not exist!`
        // });
        res.status(400).send('<script>alert(`Failed! Role ${req.body.roles[i]} does not exist!`)</script>');
        return;
      }
    }
  }

  next();
};

confirmPassword = (req, res, next) => {
  if (req.body.confirmpassword != req.body.password) {
    // res.status(400).send({
    //   message: `Failed! Password does not match its confirmation!`
    // });
    res.status(400).send('<script>alert("Failed! Password does not match its confirmation!")</script>');
    return;
  }
  next();
}

const verifySignUp = {
  checkDuplicateUsernameOrEmail,
  checkRolesExisted,
  confirmPassword
};

module.exports = verifySignUp;