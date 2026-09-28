function registerRoutes(app, dependencies) {
  const { stripe, storeItems, donationClient, dbState } = dependencies;
  const clientUrl = dependencies.clientUrl || process.env.CLIENT_URL || 'http://localhost:3000';

  app.post('/api/create-checkout-session', async (req, res) => {
    try {
      console.log(req.body.items);
      const session = await stripe.checkout.sessions.create({
        submit_type: 'donate',
        customer_email: req.body.userEmail || undefined,
        payment_method_types: ['card'],
        mode: 'payment',
        line_items: req.body.items.map(item => {
          const storeItem = storeItems.get(item.id);
          return {
            price_data: {
              currency: 'aud',
              product_data: {
                name: storeItem.name,
              },
              unit_amount: item.quantity * 100,
            },
            quantity: 1,
          };
        }),
        metadata: {
          type: 'One-Time'
        },
        success_url: `${clientUrl}/donation-success?session_id={CHECKOUT_SESSION_ID}`,
        cancel_url: clientUrl
      });
      console.log('two');
      res.json({ url: session.url });
    } catch (e) {
      console.error('Error creating Stripe checkout session:', e);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.get('/donation-success', async (req, res) => {
    const session_id = req.query.session_id;

    try {
      const session = await stripe.checkout.sessions.retrieve(session_id);
      console.log('🔍 Stripe session details:', session.customer_details);

      const donation = {
        paymentId: session.payment_intent,
        name: session.customer_details?.name || 'Anonymous',
        email: session.customer_details?.email || 'N/A',
        amount: session.amount_total / 100,
        currency: session.currency,
        method: 'Stripe',
        status: 'succeeded',
        timestamp: new Date(),
        type: session.metadata?.type || 'Unknown'
      };

      if (!dbState.connectedDB) {
        try {
          await donationClient.connect();
          dbState.connectedDB = donationClient.db('EchoNet');
          console.log('✅ Reconnected to MongoDB in fallback.');
        } catch (err) {
          console.error('❌ Retry DB connection failed:', err.message);
          return res.status(500).send('Database connection error.');
        }
      }

      const donations = dbState.connectedDB.collection('donations');
      await donations.insertOne(donation);

      console.log('✅ Stripe donation saved (via redirect)');
      res.send('<script>alert("Donation successful! Thank you."); window.location.href = "/";</script>');

    } catch (err) {
      console.error('❌ Error saving Stripe donation:', err.message);
      res.status(500).send('Error retrieving session details.');
    }
  });

  app.get('/donations', async (req, res) => {
    try {
      if (!dbState.connectedDB) {
        await donationClient.connect();
        dbState.connectedDB = donationClient.db('EchoNet');
        console.log('✅ Reconnected to MongoDB for /donations');
      }

      const donations = await dbState.connectedDB.collection('donations').find({}).toArray();
      res.json({ charges: { data: donations } });
    } catch (error) {
      console.error('❌ Error fetching donations from MongoDB:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });

  app.get('/cumulativeDonations', async (req, res) => {
    try {
      if (!dbState.connectedDB) {
        await donationClient.connect();
        dbState.connectedDB = donationClient.db('EchoNet');
      }

      const donations = await dbState.connectedDB
        .collection('donations')
        .find({ status: 'succeeded' })
        .toArray();

      const totalAmount = donations.reduce((sum, donation) => sum + (donation.amount || 0), 0);

      res.json({ cumulativeTotal: totalAmount.toFixed(2) });
    } catch (error) {
      console.error('❌ Error calculating cumulative donations:', error);
      res.status(500).json({ error: 'Internal server error' });
    }
  });
}

module.exports = { registerRoutes };
