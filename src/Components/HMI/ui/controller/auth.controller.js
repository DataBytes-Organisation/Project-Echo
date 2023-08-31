const config = require("../config/auth.config");
const db = require("../model");
const path = require('path');
const User = db.user;
const Role = db.role;

var jwt = require("jsonwebtoken");
var bcrypt = require("bcryptjs");

exports.signup = (req, res) => {
  const user = new User({
    username: req.body.username,
    email: req.body.email,
    password: bcrypt.hashSync(req.body.password, 8),
  });

  user.save((err, user) => {
    if (err) {
      res.status(500).send({ message: err });
      return;
    }

    if (req.body.roles) {
      Role.find(
        {
          name: { $in: req.body.roles },
        },
        (err, roles) => {
          if (err) {
            res.status(500).send({ message: err });
            return;
          }

          user.roles = roles.map((role) => role._id);
          user.save((err) => {
            if (err) {
              res.status(500).send({ message: err });
              return;
            }

            res.send({ message: "User was registered successfully!" });
          });
        }
      );
    } else {
      Role.findOne({ name: "user" }, (err, role) => {
        if (err) {
          res.status(500).send({ message: err });
          return;
        }

        user.roles = [role._id];
        user.save((err) => {
          if (err) {
            res.status(500).send({ message: err });
            return;
          }

          res.send({ message: "User was registered successfully!" });
        });
      });
    }
  });
};

exports.signin = (req, res) => {
  User.findOne({
    username: req.body.username,
  })
    .populate("roles", "-__v")
    .exec((err, user) => {
      if (err) {
        res.status(500).send({ message: err });
        return;
      }

      if (!user) {
        return res.status(404).send({ message: "User Not found." });
      }

      var passwordIsValid = bcrypt.compareSync(
        req.body.password,
        user.password
      );

      if (!passwordIsValid) {
        return res.status(401).send({ message: "Invalid Password!" });
      }

      const token = jwt.sign({ id: user.id },
                              config.secret,
                              {
                                algorithm: 'HS256',
                                allowInsecureKeySizes: true,
                                expiresIn: 86400, // 24 hours
                              });

      var authorities = [];

      for (let i = 0; i < user.roles.length; i++) {
        authorities.push("ROLE_" + user.roles[i].name.toUpperCase());
      }

      req.session.token = token;

      // res.status(200).send({
      //   id: user._id,
      //   username: user.username,
      //   email: user.email,
      //   roles: authorities,
      // });
      let result = {
        id: user._id,
        username: user.username,
        email: user.email,
        roles: authorities,
      }
      console.log("user login successfully: ", result);
      res.status(200).redirect("/")
    });
};

exports.signout = async (req, res) => {
  try {
    req.session = null;
    return res.status(200).send({ message: "You've been signed out!" });
  } catch (err) {
    this.next(err);
  }
};

exports.loadForget = (req,res) => {
  try {
    console.log("Sending forgetPass.html from router");
    console.log("Current dirPath: ", __dirname);
    res.status(200).sendFile(path.join(__dirname, '../public/forgetPass.html'));
  } catch (error) {
    console.log(error.message);
  }
}

exports.verifyForget = (req,res)=>{
  try {
    
    const emailToCheck = req.body.email;
    const userData = User.findOne({email:emailToCheck});
    if (userData) {
      res.send("Success");
      // res.getElementById("demo").innerHTML = "Success";
    }
    else{
      const errorMessage = "User not found.";
      res.send(errorMessage);
      //res.getElementById("demo").innerHTML = "Failed";
    }

  } catch (error) {
    console.log(error.message);
  }
}