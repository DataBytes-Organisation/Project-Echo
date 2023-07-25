const jwt = require("jsonwebtoken");
const config = require("../config/auth.config.js");
const db = require("../model");
const User = db.user;
const Role = db.role;
const path = require('path');

exports.verifyToken = (req, res, next) => {
  let token = req.session.token;
  console.log("Current token in console.log: ", token)
  if (!token) {
    // return res.status(403).send({ message: "No token provided!" });
    return res.sendFile(path.join(__dirname, '../public/login.html'));
  }

  jwt.verify(token,
            config.secret,
            (err, decoded) => {
              if (err) {
                return res.status(401).send({
                  message: "Unauthorized!",
                });
              }
              req.userId = decoded.id;
              next();
            });
};

exports.isAdmin = (req, res, next) => {
  User.findById(req.userId).exec((err, user) => {
    if (err) {
      res.status(500).send({ message: err });
      return false;
    }

    Role.find(
      {
        _id: { $in: user.roles },
      },
      (err, roles) => {
        if (err) {
          res.status(500).send({ message: err });
          return false;
        }

        for (let i = 0; i < roles.length; i++) {
          if (roles[i].name === "admin") {
            next();
            return true;
          }
        }

        res.status(403).send({ message: "Require Admin Role!" });
        return false;
      }
    );
  });
};

exports.isModerator = (req, res, next) => {
  User.findById(req.userId).exec((err, user) => {
    if (err) {
      res.status(500).send({ message: err });
      return false;
    }

    Role.find(
      {
        _id: { $in: user.roles },
      },
      (err, roles) => {
        if (err) {
          res.status(500).send({ message: err });
          return false;
        }

        for (let i = 0; i < roles.length; i++) {
          if (roles[i].name === "moderator" | roles[i].name === "admin") {
            next();
            return true;
          }
        }

        res.status(403).send({ message: "Require Moderator Role!" });
        return false;
      }
    );
  });
};

exports.isUser = (req, res, next) => {
  User.findById(req.userId).exec((err, user) => {
    if (err) {
      res.status(500).send({ message: err });
      return false;
    }

    Role.find(
      {
        _id: { $in: user.roles },
      },
      (err, roles) => {
        if (err) {
          res.status(500).send({ message: err });
          return false;
        }

        for (let i = 0; i < roles.length; i++) {
          if (roles[i].name === "moderator" | roles[i].name === "admin" | roles[i].name === 'user') {
            next();
            return true;
          }
        }

        res.status(403).send({ message: "Require User Role!" });
        return false;
      }
    );
  });
};

// const authJwt = {
//   verifyToken,
//   isAdmin,
//   isModerator,
// };
// module.exports = authJwt;