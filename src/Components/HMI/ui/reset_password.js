// Password Reset Script for TN-HMI User
const bcrypt = require('bcryptjs');
const { MongoClient } = require('mongodb');
const readline = require('readline');

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

async function resetPassword() {
    const mongoUri = "mongodb://root:root_password@localhost:27017";
    const client = new MongoClient(mongoUri);
    
    try {
        console.log("\n=== Project Echo Password Reset Tool ===\n");
        
        console.log("Connecting to MongoDB...");
        await client.connect();
        console.log("Connected successfully!");
        
        const db = client.db('UserSample');
        const usersCollection = db.collection('users');
        
        // Find TN-HMI user
        const user = await usersCollection.findOne({ username: "TN-HMI" });
        
        if (!user) {
            console.log("❌ User TN-HMI not found!");
            return;
        }
        
        console.log("\n✓ User found:");
        console.log("  Username:", user.username);
        console.log("  Email:", user.email);
        
        rl.question('\nEnter new password: ', async (newPassword) => {
            if (!newPassword || newPassword.length < 6) {
                console.log("❌ Password must be at least 6 characters!");
                rl.close();
                await client.close();
                return;
            }
            
            // Hash the new password
            console.log("\nHashing password...");
            const newPasswordHash = await bcrypt.hash(newPassword, 8);
            
            // Update the password
            await usersCollection.updateOne(
                { username: "TN-HMI" },
                { $set: { password: newPasswordHash } }
            );
            
            console.log("\n✓✓✓ Password reset successfully! ✓✓✓");
            console.log("\nYou can now login with:");
            console.log("  Username: TN-HMI");
            console.log("  Password:", newPassword);
            console.log("\nGo to: http://localhost:8080/login");
            
            rl.close();
            await client.close();
        });
        
    } catch (error) {
        console.error("❌ Error:", error.message);
        rl.close();
        await client.close();
    }
}

resetPassword();
