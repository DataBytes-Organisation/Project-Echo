const { MongoClient, ObjectId } = require('mongodb');

// Use a single shared MongoDB client
const client = new MongoClient("mongodb://localhost:27017");
const dbName = "EchoNet";

// Connect once, reuse across all routes
async function connectDb() {
  if (!client.topology || !client.topology.isConnected()) {
    await client.connect();
    console.log("🔁 Connected to MongoDB (controller)");
  }
  return client.db(dbName);
}

// Fetch all enquiries
exports.getEnquiries = async (req, res) => {
  try {
    const db = await connectDb();
    const enquiries = await db.collection("enquiries").find({}).toArray();
    res.status(200).json(enquiries);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to fetch enquiries" });
  }
};

// Create a new enquiry
exports.createEnquiry = async (req, res) => {
  try {
    const { name, email, status, notes, query } = req.body;

    // Validate incoming data
    if (!name || !email || !status || !query) {
      return res.status(400).json({ error: "Name, email, status, and query are required" });
    }

    const db = await connectDb();

    const newEnquiry = {
      name,
      email,
      status,
      notes: notes || "",
      query,
      createdAt: new Date(),
    };

    // Insert the new enquiry into the database
    const result = await db.collection("enquiries").insertOne(newEnquiry);

    res.status(201).json({
      message: "Enquiry created successfully",
      enquiry: result.ops?.[0] || newEnquiry,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to create enquiry" });
  }
};

// Update an enquiry's status and notes
exports.updateEnquiry = async (req, res) => {
  try {
    const { status, notes } = req.body;
    const id = req.params.id;

    // Validate incoming data
    if (!status || !notes) return res.status(400).json({ error: "Status and notes are required" });
    if (!ObjectId.isValid(id)) return res.status(400).json({ error: "Invalid enquiry ID" });

    const db = await connectDb();

    // Update the enquiry in the database
    const result = await db.collection("enquiries").updateOne(
      { _id: new ObjectId(id) },
      { $set: { status, notes } }
    );

    if (result.modifiedCount === 0) {
      return res.status(404).json({ error: "Enquiry not found" });
    }

    res.status(200).json({ message: "Enquiry updated successfully" });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to update enquiry" });
  }
};

// Delete an enquiry
exports.deleteEnquiry = async (req, res) => {
  try {
    const id = req.params.id;
    if (!ObjectId.isValid(id)) return res.status(400).json({ error: "Invalid enquiry ID" });

    const db = await connectDb();

    // Delete the enquiry from the database
    const result = await db.collection("enquiries").deleteOne({ _id: new ObjectId(id) });

    if (result.deletedCount === 0) {
      return res.status(404).json({ error: "Enquiry not found" });
    }

    res.status(200).json({ message: "Enquiry deleted successfully" });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to delete enquiry" });
  }
};
