// Test script to verify login credentials and MongoDB connection
const bcrypt = require('bcryptjs');
const { MongoClient } = require('mongodb');

async function testLogin() {
    const mongoUri = "mongodb://root:root_password@localhost:27017";
    const client = new MongoClient(mongoUri);
    
    try {
        console.log("Connecting to MongoDB...");
        await client.connect();
        console.log("Connected successfully!");
        
        const db = client.db('test');
        const usersCollection = db.collection('users');
        
        // Find TN-HMI user
        console.log("\nSearching for TN-HMI user...");
        const user = await usersCollection.findOne({ username: "TN-HMI" });
        
        if (!user) {
            console.log("User TN-HMI not found!");
            return;
        }
        
        console.log("\nUser found:");
        console.log("Username:", user.username);
        console.log("Email:", user.email);
        console.log("Password hash:", user.password);
        
        // Test password
        console.log("\nTesting common passwords...");
        const testPasswords = ['password', 'Password123', 'admin', 'TN-HMI', 'password123', '123456'];
        
        for (const pw of testPasswords) {
            const isValid = await bcrypt.compare(pw, user.password);
            if (isValid) {
                console.log(`✓ PASSWORD FOUND: "${pw}"`);
                return;
            } else {
                console.log(`✗ Not: "${pw}"`);
            }
        }
        
        console.log("\n⚠ No matching password found from test list.");
        console.log("If you know the password, test it with:");
        console.log(`bcrypt.compare('your_password', '${user.password}')`);
        
    } catch (error) {
        console.error("Error:", error.message);
    } finally {
        await client.close();
        console.log("\nConnection closed.");
    }
}

testLogin();
