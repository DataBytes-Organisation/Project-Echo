const express = require('express');
const path = require('path');
const axios = require('axios');
const Redis = require('ioredis');
const cors = require('cors'); // ✅ ADDED

const { MongoClient, ObjectId } = require('mongodb');

// Initialize express app
const app = express();
const port = 8080;

// Initialize Redis client
const client = new Redis(); // Adjust config if needed

// ✅ Enable CORS for all origins
app.use(cors());

// Serving static files from 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Body parser for JSON requests
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Mounting enquiries routes
app.use('/api/enquiries', require('./routes/enquiries.routes'));

// API endpoint to patch the conservation status of an animal
app.patch('/api/updateConservationStatus/:animal', async (req, res) => {
  const requestAnimal = req.params.animal;
  const newStatus = req.body.status;
  let schema = { requestAnimal, newStatus };

  let token = await client.get('JWT', (err, storedToken) => {
    if (err) {
      console.error('Error retrieving token from Redis:', err);
      return null;
    } else {
      console.log('Stored Token:', storedToken);
      return storedToken;
    }
  });

  try {
    console.log("Admin update species data: ", JSON.stringify(schema));
    const axiosResponse = await axios.patch('http://ts-api-cont:9000/hmi/api/updateConservationStatus', JSON.stringify(schema), {
      headers: { "Authorization": `Bearer ${token}`, 'Content-Type': 'application/json' }
    });

    if (axiosResponse.status === 200) {
      res.status(200).send(`<script> window.location.href = "/login"; alert("Species Data updated successfully");</script>`);
    } else {
      res.status(400).send(`<script> window.location.href = "/login"; alert("Ooops! Something went wrong with updating species data");</script>`);
    }
  } catch (error) {
    console.error(error.data);
    res.status(500).send({ error: 'Error updating species status' });
  }
});

// Fetch the requests for the admin dashboard
app.get('/api/requests', async (req, res) => {
  try {
    let token = await client.get('JWT', (err, storedToken) => {
      if (err) {
        console.error('Error retrieving token from Redis:', err);
        return null;
      } else {
        console.log('Stored Token:', storedToken);
        return storedToken;
      }
    });

    const axiosResponse = await axios.get('http://ts-api-cont:9000/hmi/requests', {
      headers: { "Authorization": `Bearer ${token}` }
    });

    if (axiosResponse.status === 200) {
      res.json(axiosResponse.data);
    } else {
      res.status(500).json({ error: 'Error fetching data' });
    }
  } catch (err) {
    console.log('Requests error: ', err);
    res.status(401).redirect('/admin-dashboard');
  }
});

// Page direction to Welcome page after logging in
app.get("/welcome", async (req, res) => {
  try {
    console.log("token: ", await client.get('JWT', (err, storedToken) => {
      if (err) {
        return `Error retrieving token from Redis: ${err}`;
      } else {
        return storedToken;
      }
    }));
    
    let role = await client.get('Roles', (err, storedToken) => {
      if (err) {
        return `Error retrieving user role from Redis: ${err}`;
      } else {
        return storedToken;
      }
    });

    if (role.toLowerCase().includes("admin")) {
      res.redirect("/admin-dashboard");
    } else {
      res.redirect("/map");
    }
  } catch {
    res.send(`<script> alert("No user info detected! Please login again"); window.location.href = "/login"; </script>`);
  }
});

// MongoDB connection URI
const uri = process.env.MONGO_URI || "mongodb://localhost:27017";

// Function to suspend or block a user
const suspendOrBlockUser = async (identifier, action) => {
  const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });

  try {
    await client.connect();
    const userdb = client.db('UserSample');
    const usersCollection = userdb.collection('users');

    const query = ObjectId.isValid(identifier)
      ? { _id: new ObjectId(identifier) }
      : { email: identifier };

    const status = action === "suspend" ? "suspended" : action === "block" ? "banned" : null;
    if (!status) throw new Error("Invalid action. Use 'suspend' or 'block'.");

    const message = "Your account has been " + status + ". Please contact support to unblock your account.";

    const result = await usersCollection.updateOne(
      query,
      { $set: { status: status, blockMessage: message } }
    );

    if (result.matchedCount === 0) {
      console.log("User not found.");
      return { success: false, message: "User not found." };
    }

    console.log(`User with ${ObjectId.isValid(identifier) ? 'ID' : 'email'} ${identifier} has been ${status}.`);
    return { success: true, message: `User has been ${status}.` };
  } catch (error) {
    console.error("Error suspending or blocking user:", error);
    return { success: false, message: "Internal server error." };
  } finally {
    await client.close();
  }
};

app.post('/suspendUser', async (req, res) => {
  const { identifier, action } = req.body;
  const result = await suspendOrBlockUser(identifier, action);
  res.status(result.success ? 200 : 500).json(result);
});

// Page direction to the map
app.get("/map", async (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start the server
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
