const jwt = require('jsonwebtoken');
const { MongoClient, ObjectId } = require('mongodb');

function registerRoutes(app, dependencies) {
  const { apiClient, redisClient, User } = dependencies;

  function isAdmin(req, res, next) {
    const token = req.headers.authorization.split(' ')[1];
    const decoded = jwt.verify(token, process.env.JWT_SECRET);

    if (decoded.role !== 'admin') {
      return res.status(403).json({ message: 'Access denied: Admins only' });
    }

    next();
  }

  app.patch('/api/users/:id/suspend', isAdmin, async (req, res) => {
    const userId = req.params.id;

    try {
      const user = await User.findByIdAndUpdate(userId, { status: 'suspended' }, { new: true });
      res.json({ message: `User ${user.email} suspended`, user });
    } catch (error) {
      res.status(500).json({ error: 'Error suspending user' });
    }
  });

  app.patch('/api/users/:id/ban', isAdmin, async (req, res) => {
    const userId = req.params.id;

    try {
      const user = await User.findByIdAndUpdate(userId, { status: 'banned' }, { new: true });
      res.json({ message: `User ${user.email} banned`, user });
    } catch (error) {
      res.status(500).json({ error: 'Error banning user' });
    }
  });

  app.patch('/api/users/:id/reinstate', isAdmin, async (req, res) => {
    const userId = req.params.id;

    try {
      const user = await User.findByIdAndUpdate(userId, { status: 'active' }, { new: true });
      res.json({ message: `User ${user.email} reinstated`, user });
    } catch (error) {
      res.status(500).json({ error: 'Error reinstating user' });
    }
  });

  app.get('/api/users/:id/status', isAdmin, async (req, res) => {
    const userId = req.params.id;

    try {
      const user = await User.findById(userId, 'email status');
      res.json({ user });
    } catch (error) {
      res.status(500).json({ error: 'Error retrieving user status' });
    }
  });

  app.post('/api/applyAlgorithm', (req, res) => {
    const { algorithm } = req.body;
    console.log(`Algorithm ${algorithm} applied.`);
    res.send(`Algorithm ${algorithm} has been applied.`);
  });

  app.post('/api/submit', async (req, res) => {
    let token = await redisClient.get('JWT', (err, storedToken) => {
      if (err) {
        console.error('Error retrieving token from Redis:', err);
        return null;
      } else {
        console.log('Stored Token:', storedToken);
        return storedToken;
      }
    });
    let schema = req.body;
    schema.status = 'pending';
    schema.date = new Date();
    try {
      console.log('Request submission data: ', JSON.stringify(schema));
      await apiClient.post('/hmi/api/submit', schema, { headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' } });
      res.status(201).send('<script> window.location.href = "/login"; alert("Request Submitted successfully");</script>');
    } catch (error) {
      console.error(error.message);
      res.status(500).send('An error occurred');
    }
  });

  app.post('/api/approve', async (req, res) => {

  });

  app.patch('/api/requests/:id', async (req, res) => {
    const requestId = req.params.id;
    const newStatus = req.body.status;
    let schema = { requestId: requestId, newStatus: newStatus };
    let token = await redisClient.get('JWT', (err, storedToken) => {
      if (err) {
        console.error('Error retrieving token from Redis:', err);
        return null;
      } else {
        console.log('Stored Token:', storedToken);
        return storedToken;
      }
    });
    try {
      console.log('Admin Request update data: ', JSON.stringify(schema));
      await apiClient.patch('/hmi/api/requests', schema, { headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' } });
      res.status(200).send('<script> window.location.href = "/login"; alert("Request data updated successfully");</script>');
    } catch (error) {
      console.error(error.message);
      res.status(500).send({ error: 'Error updating request status' });
    }
  });

  app.patch('/api/updateConservationStatus/:animal', async (req, res) => {
    const requestAnimal = req.params.animal;
    const newStatus = req.body.status;
    let schema = { requestAnimal: requestAnimal, newStatus: newStatus };
    let token = await redisClient.get('JWT', (err, storedToken) => {
      if (err) {
        console.error('Error retrieving token from Redis:', err);
        return null;
      } else {
        console.log('Stored Token:', storedToken);
        return storedToken;
      }
    });
    try {
      console.log('Admin update species data: ', JSON.stringify(schema));
      await apiClient.patch('/hmi/api/updateConservationStatus', schema, { headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' } });
      res.status(200).send('<script> window.location.href = "/login"; alert("Species Data updated successfully");</script>');
    } catch (error) {
      console.error(error.message);
      res.status(500).send({ error: 'Error updating species status' });
    }
  });

  app.get('/api/requests', async (req, res) => {
    try {
      let token = await redisClient.get('JWT', (err, storedToken) => {
        if (err) {
          console.error('Error retrieving token from Redis:', err);
          return null;
        } else {
          console.log('Stored Token:', storedToken);
          return storedToken;
        }
      });

      const data = await apiClient.get('/hmi/requests', { headers: { 'Authorization': `Bearer ${token}` } });
      res.json(data);
    } catch (err) {
      console.log('Requests error: ', err);
      res.status(401).redirect('/admin-dashboard');
    }
  });

  const uri = process.env.MONGO_URI || `mongodb://${process.env.DB_HOST || 'localhost'}:27017`;
  const suspendOrBlockUser = async (identifier, action) => {
    const client = new MongoClient(uri, { useNewUrlParser: true, useUnifiedTopology: true });

    try {
      await client.connect();
      const userdb = client.db('UserSample');
      const usersCollection = userdb.collection('users');
      const query = ObjectId.isValid(identifier)
        ? { _id: new ObjectId(identifier) }
        : { email: identifier };
      const status = action === 'suspend' ? 'suspended' : action === 'block' ? 'banned' : null;
      if (!status) {
        throw new Error('Invalid action. Use \'suspend\' or \'block\'.');
      }
      const message = 'Your account has been ' + status + '. Please contact support to unblock your account.';

      const result = await usersCollection.updateOne(
        query,
        { $set: { status: status, blockMessage: message } }
      );

      if (result.matchedCount === 0) {
        console.log('User not found.');
        return { success: false, message: 'User not found.' };
      }

      console.log(`User with ${ObjectId.isValid(identifier) ? 'ID' : 'email'} ${identifier} has been ${status}.`);
      return { success: true, message: `User has been ${status}.` };
    } catch (error) {
      console.error('Error suspending or blocking user:', error);
      return { success: false, message: 'Internal server error.' };
    } finally {
      await client.close();
    }
  };

  app.post('/suspendUser', async (req, res) => {
    const { identifier, action } = req.body;
    const result = await suspendOrBlockUser(identifier, action);
    res.status(result.success ? 200 : 500).json(result);
  });
}

module.exports = { registerRoutes };
