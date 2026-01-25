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
        
        // List all databases
        const adminDb = client.db().admin();
        const dbs = await adminDb.listDatabases();
        console.log("\nAvailable databases:");
        dbs.databases.forEach(db => console.log(`  - ${db.name}`));
        
        // Try to find users in different databases
        const dbNames = ['EchoNet', 'UserSample', 'test', 'project-echo', 'echo', 'admin', 'projectecho'];
        
        for (const dbName of dbNames) {
            console.log(`\nChecking database: ${dbName}...`);
            const db = client.db(dbName);
            const collections = await db.listCollections().toArray();
            console.log(`  Collections: ${collections.map(c => c.name).join(', ')}`);
            
            if (collections.some(c => c.name === 'users')) {
                const usersCollection = db.collection('users');
                const userCount = await usersCollection.countDocuments();
                console.log(`  Found ${userCount} users in ${dbName}.users`);
                
                // Find TN-HMI user
                const user = await usersCollection.findOne({ username: "TN-HMI" });
                
                if (user) {
                    console.log("\n✓ User TN-HMI found in database:", dbName);
                    console.log("Username:", user.username);
                    console.log("Email:", user.email);
                    console.log("Password hash:", user.password);
                    
                    // Test password
                    console.log("\nTesting common passwords...");
                    const testPasswords = [
                        'password', 'Password123', 'admin', 'TN-HMI', 'password123', '123456', 'tnhmi',
                        'deakin', 'Deakin123', 'projectecho', 'ProjectEcho', 'echo', 'Echo123',
                        'TN-HMI123', 'tnhmi123', 'test', 'Test123', '12345678', 'qwerty',
                        'Password1', 'Password@123', 'Admin123', 'admin123'
                    ];
                    
                    for (const pw of testPasswords) {
                        const isValid = await bcrypt.compare(pw, user.password);
                        if (isValid) {
                            console.log(`\n✓✓✓ PASSWORD FOUND: "${pw}" ✓✓✓`);
                            return;
                        } else {
                            console.log(`  ✗ Not: "${pw}"`);
                        }
                    }
                    
                    console.log("\n⚠ No matching password found from test list.");
                    return;
                }
            }
        }
        
        console.log("\n⚠ User TN-HMI not found in any database!");
        
    } catch (error) {
        console.error("Error:", error.message);
    } finally {
        await client.close();
        console.log("\nConnection closed.");
    }
}

testLogin();
