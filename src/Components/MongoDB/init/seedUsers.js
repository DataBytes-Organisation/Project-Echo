const mongoose = require('mongoose');
const fs = require('fs');
const path = require('path');

// Use same schema but define collection name and DB explicitly
const userSchema = new mongoose.Schema({
  username: String,
  email: String,
  password: String,
  mfaEnabled: { type: Boolean, default: false },
  roles: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Role' }],
  status: { type: String, default: 'Active' }
}, { collection: 'users' });

const User = mongoose.model('User', userSchema);

mongoose.connect('mongodb://root:root_password@localhost:27017/EchoNet?authSource=admin')
  .then(async () => {
    console.log('Connected to MongoDB');

    const filePath = path.join(__dirname, 'init', 'user-seed.json');
    const rawData = fs.readFileSync(filePath);
    const users = JSON.parse(rawData);

    await User.deleteMany(); // clear existing
    await User.insertMany(users);

    console.log('Users seeded successfully');
    mongoose.disconnect();
  })
  .catch(err => {
    console.error('Seeding error:', err);
    mongoose.disconnect();
  });