const config = require("../config/auth.config");
const db = require("../model");
const crypto = require("crypto")
const User = db.user;
const Role = db.role;
const Guest = db.guest;

var jwt = require("jsonwebtoken");
var bcrypt = require("bcryptjs");

const nodemailer = require("nodemailer");

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

    // Set up Nodemailer transporter
    const transporter = nodemailer.createTransport({
      service: "gmail",
      auth: {
        user: "echodatabytes@gmail.com",
        pass: "ltzoycrrkpeipngi",
      },
    });

    // Send registration confirmation email
    const mailOptions = {
      from: "echodatabytes@gmail.com",
      to: user.email,
      subject: "Registration Successful",
      html: `Congratulations! You have successfully registered.<hr> Dear ${user.username},<br> You've successfully been registered for the project echo. <br> You can now enter your credentials and sign in your personal account <hr>.`,
    };

    transporter.sendMail(mailOptions, (err, info) => {
      if (err) {
        console.error("Error sending email:", err);
      } else {
        console.log("Email sent:", info.response);
      }
    });

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

            console.log("User was registered successfully!");
            res.status(200).send('<script> window.location.href = "/login"; alert("User was registered successfully!");</script>');


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

          console.log("User was registered successfully!");
          res.status(200).send('<script>window.location.href = "/login"; alert("User was registered successfully!"); </script>');

        });
      });
    }
  });
};

exports.signin = (req, res) => {
  User.findOne({
    $or: [
      { username: req.body.username },
      { email: req.body.email }
    ]
  })
    .populate("roles", "-__v")
    .exec((err, user) => {
      if (err) {
        res.status(500).send({ message: err });
        return;
      }

      if (!user) {
        return res.status(404).send('<script> alert("User does not exist"); window.location.href = "/login";  </script>');
      }

      var passwordIsValid = bcrypt.compareSync(
        req.body.password,
        user.password
      );

      if (!passwordIsValid) {
        return res.status(401).send('<script> alert("Incorrect Password!"); window.location.href = "/login";  </script>');

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
      res.status(200).redirect("/welcome")
    });
};

exports.updatepassword = (req, res) => {
  User.findOne({
    $or: [
      { username: req.body.username },
      { email: req.body.email }
    ]
  })
  
}

//Guest signup, with random UserId and Password
exports.guestsignup = async (req) => {
  const guest = new Guest({
    userId: "guest_" + crypto.randomUUID().toString(),
    username: req.username,
    email: req.email,
    password: bcrypt.hashSync(req.password, 8),
    expiresAt: req.timestamp
  });
  console.log("Guest is looking: ", guest);
  console.log("unecrypted password: ", req.password)
  guest.save((err, guest) => {
    if (err) {
      // res.status(500).send({ message: err });
      console.log("Cannot created new Guest account: " + err)
      return { status: "failed", message: "Create Guest Error! Please check the console log" };
    }
    Role.findOne({ name: "guest" }, (err, role) => {
      if (err) {
        console.log("Cannot find the role document of guest: " + err)
        // res.status(500).send({ message: err });
        return { status: "failed", message: "Find Role Error! Check the console log" };
      }
      guest.roles = [role._id];
      guest.save((err) => {
        if (err) {
          console.log("Cannot save Guest: ", err)
          // res.status(500).send({ message: err });
          return { status: "failed", message: "Save Guest Error! Please check the console log" };
        }
        return { status: 'success' };
      });
    });
  }
  );
  return { status: 'success' };
};

exports.guestsignin = (req, res) => {
  Guest.findOne({
    username: req.body.username,
  })
    .populate("roles", "-__v")
    .exec((err, user) => {
      if (err) {
        res.status(500).send({ message: err });
        return;
      }

      if (!user) {
        return res.status(404).send({ message: "Guest Not found." });
      }

      var passwordIsValid = bcrypt.compareSync(
        req.body.password,
        user.password
      );

      if (!passwordIsValid) {
        return res.status(401).send({ message: "Invalid Password!" });
      }

      const token = jwt.sign({ id: user.userId },
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
      //Assign session token
      req.session.token = token;

      //Log result
      let result = {
        id: user._id,
        username: user.username,
        email: user.email,
        roles: authorities,
      }
      console.log("Guest login successfully: ", result);
      res.status(200).redirect("/welcome")
    });
};


exports.signout = async (req, res) => {
  try {
    req.session = null;
    return res.status(200).send({ message: "You've been signed out!", session: req.session });
  } catch (err) {
    this.next(err);
  }
};