const express = require("express");
const app = express();
const path = require("path");
const fs = require("fs");
const cookieSession = require("cookie-session");
const jwt = require("jsonwebtoken");
const { authJwt, client } = require("./middleware");
const controller = require("./controller/auth.controller");
const crypto = require("crypto");
const bcrypt = require("bcryptjs");
const mongoose = require("mongoose");

client.connect();

const cors = require("cors");
require("dotenv").config();
//const shop = require("./shop/shop")
const stripe = require("stripe")(process.env.STRIPE_PRIVATE_KEY);
const axios = require("axios");

const { createCaptchaSync } = require("captcha-canvas");
//Add mongoDB module inside config folder
const db = require("./model");
const Role = db.role;
const User = db.user;
const Guest = db.guest;
const Request = db.request;

const db_host = process.env.DB_HOST;
const api_host = process.env.API_HOST;
const db_user = process.env.DB_USER;
const db_user_pwd = process.env.DB_USER_PASS;
const db_root = process.env.DB_ROOT_USER;
const db_root_pwd = process.env.DB_ROOT_USER_PASS;

//for connecting to api-server
const api_url = `http://${api_host}:9000/`;

function testApi() {
  axios
    .get(api_url)
    .then((response) => {
      //here we are checking if the response from api is a string because at root (/) it returns "Welcome to echo api, move to /docs for more"
      if (typeof response.data === "string") {
        console.log("API returned a string:", response.data);
      } else {
        console.log(
          "API did not return a string. Instead, it returned:",
          response.data
        );
      }
    })
    .catch((error) => {
      console.error("Error connecting to http://localhost:9000:", error);
    });
}

testApi();

//for connecting to ts-mongo-db
const MongoClient = require("mongodb").MongoClient;
const url = `mongodb://${db_user}:${db_user_pwd}@${db_host}:27017/EchoNet`;

// Create a function to test the MongoDB connection

//Security verification for email account and body content validation:
const validation = require("deep-email-validator");
const mongoSanitize = require("express-mongo-sanitize");
db.mongoose
  .connect(
    `mongodb://${db_root}:${db_root_pwd}@${db_host}/UserSample?authSource=admin`,
    {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    }
  )
  .then(() => {
    console.log("Successfully connect to MongoDB.");
    initial();
    initUsers();
    initGuests();
    initRequests();
  })
  .catch((err) => {
    console.log(
      "ConnString: ",
      `mongodb://${db_root}:${db_root_pwd}@${db_host}/UserSample?authSource=admin`
    );
    console.error("Connection error", err);
    // process.exit();
  });

//mongoose.connect("mongodb://modelUser:EchoNetAccess2023@localhost:27017/EchoNet")
//Initalize the data if no user role existed
function initial() {
  Role.estimatedDocumentCount((err, count) => {
    if (!err && count === 0) {
      // new Role({
      //   name: "user"
      // }).save(err => {
      //   if (err) {
      //     console.log("error", err);
      //   }

      //   console.log("added 'user' to roles collection");
      // });
      // new Role({
      //   name: "admin"
      // }).save(err => {
      //   if (err) {
      //     console.log("error", err);
      //   }

      //   console.log("added 'admin' to roles collection");
      // });
      const roleData = require(path.join(
        __dirname,
        "user-sample/role-seed.json"
      ));

      Role.insertMany(roleData);
    }
  });
}

// async function getAllPayments() {

//   while (true) {
//     nextPage = null;
//     firstPage = false;
//     let charges;
//     if(firstPage == false){
//       charges = await stripe.charges.list({
//         limit: 100,
//       });
//       firstPage = true;
//     }
//     charges.data.forEach(charge => {
//       cumulativeTotal += charge.amount;
//     });
//     if (!charges.has_more) {
//       break; // Exit the loop when there are no more pages
//     }
//     nextPage = charges[charges.length() - 1]
//     charges = await stripe.charges.list({
//       limit: 100,
//       starting_next: nextPage
//     });
//     firstPage = true;
//   }
//   console.log('Cumulative Total:', cumulativeTotal);
// }

// getAllPayments();

app.get("/donations", async (req, res) => {
  let charges;
  try {
    while (true) {
      nextPage = null;
      firstPage = false;
      if (firstPage == false) {
        charges = await stripe.charges.list({
          limit: 100,
        });
        firstPage = true;
      }

      if (!charges.has_more) {
        break; // Exit the loop when there are no more pages
      }
      nextPage = charges[charges.length() - 1];
      charges = await stripe.charges.list({
        limit: 100,
        starting_next: nextPage,
      });
      firstPage = true;
    }
    res.json({ charges });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.get("/cumulativeDonations", async (req, res) => {
  let cumulativeTotal = 0;
  try {
    while (true) {
      nextPage = null;
      firstPage = false;
      let charges;
      if (firstPage == false) {
        charges = await stripe.charges.list({
          limit: 100,
        });
        firstPage = true;
      }
      charges.data.forEach((charge) => {
        cumulativeTotal += charge.amount;
      });
      if (!charges.has_more) {
        break; // Exit the loop when there are no more pages
      }
      nextPage = charges[charges.length() - 1];
      charges = await stripe.charges.list({
        limit: 100,
        starting_next: nextPage,
      });
      firstPage = true;
    }
    cumulativeTotal = cumulativeTotal / 100;
    cumulativeTotal = cumulativeTotal.toFixed(2);

    console.log("Cumulative Total:", cumulativeTotal);
    res.json({ cumulativeTotal });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

function initRequests() {
  Request.estimatedDocumentCount((err, count) => {
    if (!err && count === 0) {
      const requestData = require(path.join(
        __dirname,
        "user-sample/request-seed.json"
      ));
      Request.insertMany(requestData);
    }
  });
}
//Add sample Users if none exists
function initUsers() {
  User.estimatedDocumentCount((err, count) => {
    if (!err && count === 0) {
      const userData = require(path.join(
        __dirname,
        "user-sample/user-seed.json"
      ));
      User.insertMany(userData);
    }
  });
}

//Add sample Guest users if none exists
function initGuests() {
  Guest.estimatedDocumentCount((err, count) => {
    if (!err && count === 0) {
      //Different from Roles and Users,
      // another approach is to manually seed mongoDB document
      // Only feasible if there are only 1-2 sample documents
      const newGuest1 = new Guest({
        userId: "HMITest1",
        username: "guest_tester1_0987654321",
        email: "guest@echo.com",
        password: bcrypt.hashSync("guest_password", 8),
        roles: [
          {
            _id: "64be1d0f05225843178d91d7",
          },
        ],
        expiresAt: new Date(Date.now() + 1800000), // Set the expiration duration for 30 mins = 1800 s = 1800000 ms from now
      });

      // Save the new guest document to the collection
      newGuest1.save((err, doc) => {
        if (err) {
          console.error(err);
        } else {
          console.log("Guest document inserted successfully:", doc);
        }
      });

      const newGuest2 = new Guest({
        userId: "HMITest2",
        username: "guest_tester2_1234567890",
        email: "guest@hmi.com",
        password: bcrypt.hashSync("guest_password", 8),
        roles: [
          {
            _id: "64be1d0f05225843178d91d7",
          },
        ],
        expiresAt: new Date(Date.now() + 300000), // Set the expiration duration for 5 mins = 300 s = 300000 ms from now
      });

      // Save the new guest document to the collection
      newGuest2.save((err, doc) => {
        if (err) {
          console.error(err);
        } else {
          console.log("Guest document inserted successfully:", doc);
        }
      });
    }
  });
}
app.get("/data/captcha", (req, res) => {
  const { image, text } = createCaptchaSync(300, 100); // Use the package's functionality
  fs.writeFileSync("./public/captchaImg.png", image);
  console.log("text: ", text);
  console.log("Image: ", image);
  res.json({ image, text });
});
//Background Process to automatically delete Guest role after exceeding expiration
setInterval(() => {
  const now = new Date();
  console.log("Background monitor at ", now.toString());
  Guest.deleteMany({ expiresAt: { $lte: now } }, (err) => {
    if (err) {
      console.error("Error deleting expired documents:", err);
    } else {
      console.log("Expired documents deleted successfully.");
    }
  });
}, 360000); // Run every 6 mins = 360 s = 360000 ms (adjust as needed)

const port = 8080;

// serve static files from the public directory
app.use(express.static(path.join(__dirname, "public")));

var corsOptions = {
  origin: ["http://localhost:8081", "*"],
};

app.use(cors(corsOptions));

//bodyParser to make sure post form data is read
const bodyParser = require("express");
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

//MongoDB query sanitization
//Run in dryRun = testing mode; prevent server interruption because of this process
app.use(
  mongoSanitize({
    dryRun: true,
    onSanitize: ({ req, key }) => {
      console.warn(`[DryRun] This request[${key}] will be sanitized`, req);
    },
  })
);

//const serveIndex = require('serve-index');
//app.use('/images/bio', serveIndex(express.static(path.join(__dirname, '/images/bio'))));

app.use(
  cookieSession({
    name: "echo-session",
    keys: ["COOKIE_SECRET"], // should use as secret environment variable
    httpOnly: true,
  })
);

const nodemailer = require("nodemailer");
const { verifyToken } = require("./middleware/authJwt");
var transporter = nodemailer.createTransport({
  service: "gmail",
  auth: {
    user: "echodatabytes@gmail.com",
    pass: "ltzoycrrkpeipngi",
  },
});

// Function to escape special characters to HTML entities
function escapeHtmlEntities(input) {
  return input.replace(/[\u00A0-\u9999<>&]/gim, function (i) {
    return "&#" + i.charCodeAt(0) + ";";
  });
}

async function testEmail(input) {
  let res = await validation.validate(input);
  return { result: res.valid, response: res.validators };
}

app.post("/send_email", async (req, res) => {
  const { email, query } = req.body;
  const validationResult = await testEmail(email);
  if (validationResult.result) {
    // Validate the email address
    // const validationResult = await validateEmail(email);
    // console.log("Validate email result: ", validationResult);
    // If email validation is successful, proceed to send the email
    let html_text = "<div>";
    html_text += "<h2>A new query has been received for Project Echo HMI</h2>";
    html_text +=
      '<img src="cid:logo@echo.hmi" style="height: 150px; width: 150px; display: flex; margin: auto;"/>';
    html_text += "<p>Sender: \t " + email + "</p>"; // Convert sender's email to HTML entities
    html_text += "<p>Query: \t " + escapeHtmlEntities(query) + "</p>"; // Convert query to HTML entities
    html_text += "<hr>";
    html_text +=
      "<p>Yes, this mailbox is active. So please feel free to reply to this email if you have other queries.</p>";
    html_text += "</div>";

    let mailOptions = {
      from: email,
      to: `echodatabytes@gmail.com, ${email}, databytes@deakin.edu.au`,
      subject: "New query received!",
      text: query,
      html: html_text,
      attachments: [
        {
          // stream as an attachment
          filename: "image.png",
          content: fs.createReadStream(
            path.join(__dirname, "public/images/tabIcons/logo.png")
          ),
          cid: "logo@echo.hmi", //same cid value as in the html
        },
      ],
    };

    transporter.sendMail(mailOptions, function (error, info) {
      if (error) {
        console.log(error);
      } else {
        console.log("Email sent: " + info.response);
        return res.redirect("/");
      }
    });
  } else {
    return res.status(400).json({ error: validationResult.response });
  }
});

app.get("/send_email", (req, res) => {
  setTimeout(() => res.redirect("/"), 5000);
});

var chars =
  "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^&*()ABCDEFGHIJKLMNOPQRSTUVWXYZ";

function genPass(length) {
  let password = "";
  for (var i = 0; i <= parseInt(length); i++) {
    var randomNumber = Math.floor(Math.random() * chars.length);
    password += chars.substring(randomNumber, randomNumber + 1);
  }
  return password;
}

app.post("/request_access", async (req, res) => {
  console.log("email: ", req.body.email);
  const { email } = req.body;
  //Generate Guest credentials + timestamp
  let salt = "";
  while (salt.length < 8) {
    salt = crypto.getRandomValues(new Uint32Array(1)).toString();
  }
  let username = "guest_" + email.split("@")[0] + "_" + salt;
  console.log("username: ", username);
  let password = genPass(12);
  let timestamp = new Date(Date.now() + 1800000); //Set time to live of 1800000 ms = 1800 s = 30 mins
  let request = {
    username: username,
    email: req.body.email,
    password: password,
    timestamp: timestamp,
  };
  try {
    //Sending that to Guest signup
    const response = await controller.guestsignup(request);

    setTimeout(() => {
      console.log("response is back! ", response);
      //Send email to user when success
      if (response && response.status === "success") {
        let html_text = "<div>";
        html_text += "<h2>Echo HMI Temporary Access Requested!</h2>";
        html_text +=
          '<img src="cid:logo@echo.hmi" style="height: 150px; width: 150px; display: flex; margin: auto;"/>';
        html_text += "<p>Dear \t <strong>" + req.body.email + "</strong></p>";
        html_text += "<hr>";
        html_text +=
          "<p>Thank you for your patience, here is your login credential </p>";
        html_text += "<p><strong>Username:</strong> \t " + username + "</p>";
        html_text += "<p><strong>Password:</strong> \t " + password + "</p>";
        html_text +=
          "<br><p>Please take in mind that this account will only be valid until " +
          timestamp.toString() +
          " (Subject to change based on development)</p>";
        html_text += "</div>";
        let mailOptions = {
          from: email,
          to: `echodatabytes@gmail.com, ${email}`,
          subject: "Guest User Access Granted!",
          html: html_text,
          attachments: [
            {
              // stream as an attachment
              filename: "image.png",
              content: fs.createReadStream(
                path.join(__dirname, "public/images/tabIcons/logo.png")
              ),
              cid: "logo@echo.hmi", //same cid value as in the html
            },
          ],
        };
        transporter.sendMail(mailOptions, function (error, info) {
          if (error) {
            console.log(error);
          } else {
            console.log("Email sent: " + info.response);
            return res.redirect("/login");
          }
        });
      } else {
        console.log("Something happened for Guest Access Granting: ", response);
        let error_box = document.getElementById("request-access-email-error");
        error_box.innerHTML = `Exception error occured: ${response.message}`;
        error_box.style.display = "block";
        setTimeout(() => {
          error_box.innerHTML = "";
          error_box.style.display = "none";
        }, 3000);
      }
    }, 200);
  } catch (error) {
    res.status(500).send({
      message: "An error occurred while sending the request access: " + error,
    });
  }
});

// routes
require("./routes/auth.routes")(app);
require("./routes/user.routes")(app);

app.get("/", (req, res) => {
  if (authJwt.verifyToken && authJwt.isUser) {
    console.log("This is user session!");
    res.sendFile(path.join(__dirname, "public/index.html"));
  } else {
    console.log("This is not user sessions!");
    res.json("No available session, you have not logged in yet");
  }
});

app.get("/admin-dashboard", (req, res) => {
  return res.sendFile(path.join(__dirname, "public/admin/dashboard.html"));
});

app.get("/admin-template", (req, res) => {
  return res.sendFile(path.join(__dirname, "public/admin/template.html"));
});

app.get("/admin-donations", (req, res) => {
  return res.sendFile(path.join(__dirname, "public/admin/donations.html"));
});

app.get("/login", (req, res) => {
  [verifyToken];
  res.sendFile(path.join(__dirname, "public/login.html"));
});

app.post("/api/submit", async (req, res) => {
  try {
    const newRequest = new Request(req.body);
    newRequest.date = new Date();
    newRequest.status = "pending";
    await newRequest.save();
    res.status(200).send("Request submitted successfully");
  } catch (error) {
    console.error(error);
    res.status(500).send("An error occurred");
  }
});

async function testMongoDBConnection() {
  try {
    // Attempt to connect to MongoDB
    const client = new MongoClient(url, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    await client.connect();

    // If the connection is successful, print a success message
    console.log("MongoDB connection test: Connection successful");

    // Perform additional database operations here if needed

    // Close the connection when done
    await client.close();
  } catch (error) {
    // If there's an error, print an error message
    console.error("MongoDB connection test: Connection failed");
    console.error(error);
  }
}

// Call the function to test the MongoDB connection
testMongoDBConnection();

app.post("/api/approve", async (req, res) => {});

app.get("/requests", (req, res) => {
  res.sendFile(path.join(__dirname, "public/admin/admin-request.html"));
});

app.get("/requestsOriginal", (req, res) => {
  res.sendFile(path.join(__dirname, "public/requests.html"));
});

app.patch("/api/requests/:id", async (req, res) => {
  const requestId = req.params.id; // Get the request ID from the URL parameter
  const newStatus = req.body.status; // Get the new status from the request body

  try {
    // Find the request by ID and update the status
    const updatedRequest = await Request.findByIdAndUpdate(
      requestId,
      { $set: { status: newStatus } },
      { new: true } // Return the updated document
    );

    if (!updatedRequest) {
      return res.status(404).json({ error: "Request not found" });
    }

    res.json({
      message: "Request status updated successfully",
      updatedRequest,
    });
  } catch (error) {
    console.error("Error updating request status:", error);
    res.status(500).json({ error: "Error updating request status" });
  }
});

app.patch("/api/updateConservationStatus/:animal", async (req, res) => {
  const requestAnimal = req.params.animal;
  const newStatus = req.body.status;
  try {
    const client = new MongoClient(url, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    await client.connect();
    const EchoNet = client.db();
    const collection = EchoNet.collection("species");
    //const query = { _id : requestAnimal };
    //const updateOperation = {status : newStatus};
    const result = await collection.findOneAndUpdate(
      { _id: requestAnimal },
      { $set: { status: newStatus } },
      {
        collation: { locale: "en", strength: 2 }, // Case-insensitive collation
        returnOriginal: false, // Set this to false to get the updated document
      }
    );
    if (result.value) {
      // If a matching document is found and updated, print it to the console
      console.log("Updated animal:", result.value);
    } else {
      // If no matching document is found, print a message
      console.log("Animal not found.");
    }
    client.close();
    res.status(200).json({
      message: `updated animal status successfully ${result.value}, ${requestAnimal}, ${newStatus}`,
    });
  } catch (error) {
    console.error("MongoDB connection or update operation failed:", error);
    res.status(500).json({ error: "Error updating animal" });
  }
});

// OLD METHOD - USING DIRECT CONNECTION
// app.get('/api/requests', async (req, res) => {
//   try {
//     const requests = await Request.find();
//     res.json(requests);
//   } catch (error) {
//     res.status(500).json({ error: 'Error fetching data' });
//   }
// });

// NEW METHOD - CONNECT VIA API
app.get("/api/requests", async (req, res) => {
  try {
    let token = await client.get("JWT", (err, storedToken) => {
      if (err) {
        console.error("Error retrieving token from Redis:", err);
        return null;
      } else {
        console.log("Stored Token:", storedToken);
        return storedToken;
      }
    });

    console.log("Current token: ", token);
    const axiosResponse = await axios.get(
      "http://localhost:9000/hmi/requests",
      { headers: { Authorization: `Bearer ${token}` } }
    );

    if (axiosResponse.status === 200) {
      res.json(axiosResponse.data);
    } else {
      res.status(500).json({ error: "Error fetching data" });
    }
  } catch (err) {
    console.log("Requests error: ", err);
    res.status(401).redirect("/admin-dashboard");
  }
});

app.get("/welcome", async (req, res) => {
  console.log(
    "token: ",
    await client.get("JWT", (err, storedToken) => {
      if (err) {
        console.error("Error retrieving token from Redis:", err);
        return null;
      } else {
        console.log("Stored Token:", storedToken);
        return storedToken;
      }
    })
  );
  res.sendFile(path.join(__dirname, "public/index.html"));
});

// app.get("*", (req,res) => {
//   if (authJwt.verifyToken){
//     let token = req.session.token;
//     console.log("Current token: ", req.session.token)
//     if (!token) {
//       console.log("Current user session unavailable")
//       res.sendFile(path.join(__dirname, 'public/login.html'));
//     }
//   }
//   else {
//     console.log("User token not assigned!");
//     res.sendFile(path.join(__dirname, 'public/login.html'));
//   }

// })

// start the server
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
