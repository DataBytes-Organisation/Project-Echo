const express = require("express");
const mongoose = require("mongoose");
const passport = require("passport");
const LocalStrategy = require("passport-local").Strategy;
const session = require("express-session");
const app = express();
const path = require("path");
const base = `${__dirname}/web`;
const port = 5001;

require('dotenv').config(); // Load variables from .env file

const secretKey = process.env.MY_APP_SECRET_KEY;
// ... (use secretKey where needed)

app.use(express.static('web'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));


// Initialize Passport
app.use(passport.initialize());


// Configure session
app.use(session({
  secret: secretKey, // Replace with a secure secret key
  resave: false,
  saveUninitialized: false
}));

app.use(passport.session());

uri = "mongodb+srv://dipansh:qwerty99@mydb.chvnp5g.mongodb.net/?retryWrites=true&w=majority";

const userSchema = new mongoose.Schema({
  email: String,
  username: String,
  password: String,
});

const User = mongoose.model("User", userSchema);

async function connect() {
  try {
    await mongoose.connect(uri);
    console.log("Connected to MongoDB");
  } catch (error) {
    console.error(error);
  }
}

connect();

passport.use(new LocalStrategy({
  usernameField: 'identifier', // This field should match the name in the login request
  passwordField: 'password'
},
async (identifier, password, done) => {
  try {
    const user = await User.findOne({
      $or: [{ email: identifier }, { username: identifier }],
      password: password
    });

    if (user) {
      return done(null, user);
    } else {
      return done(null, false, { message: "Invalid credentials" });
    }
  } catch (error) {
    return done(error);
  }
}
));

// Serialize user for session
passport.serializeUser((user, done) => {
done(null, user.id);
});

// Deserialize user
passport.deserializeUser(async (id, done) => {
try {
  const user = await User.findById(id);
  done(null, user);
} catch (error) {
  done(error);
}
});

app.post("/login", async (req, res) => {
  const { identifier, password } = req.body;

  try {
    const user = await User.findOne({
      $or: [{ email: identifier }, { username: identifier }],
      password: password
    });

    if (user) {
      res.json({
        message: "Login successful",
        redirectTo: "/welcome" // Redirect to welcome.html on success
      });
      console.log("Successful login")
    } else {
      res.status(401).json({ error: "Invalid credentials" });
    }
  } catch (error) {
    res.status(500).json({ error: "Internal server error" });
  }
});

app.post("/signup", async (req, res) => {
  const { username, email, password } = req.body;

  try {
    // Check if the email or username is already registered
    const existingUser = await User.findOne({ $or: [{ email }, { username }] });
    if (existingUser) {
      return res.status(409).json({ error: "Email or username already exists" });
    }

    // Create a new user
    const newUser = new User({ username, email, password });
    await newUser.save();

    // Sign up successful
    res.json({ message: "Sign up successful" });
  } catch (error) {
    res.status(500).json({ error: "Internal server error" });
  }
});

app.get("/", (req, res) => {
  res.sendFile(`${base}/index.html`);
});

app.get("/signup", (req, res) => {
  res.sendFile(`${base}/sign_up.html`);
});

app.get("/welcome", (req, res) => {
  res.sendFile(`${base}/welcome.html`);
});

app.listen(port, () => {
  console.log('Server is running on port:', port);
});
