// routes/payment.routes.js
const express = require('express');
const router = express.Router();
const stripe = require('stripe')(process.env.STRIPE_PRIVATE_KEY);

const storeItems = new Map([[1, { priceInCents: 100, name: "donation" }]]);

router.post("/create-checkout-session", async (req, res) => {
  try {
    const session = await stripe.checkout.sessions.create({
      submit_type: 'donate',
      customer_email: req.body.userEmail,
      payment_method_types: ["card"],
      mode: "payment",
      line_items: req.body.items.map(item => {
        const storeItem = storeItems.get(item.id);
        return {
          price_data: {
            currency: "aud",
            product_data: { name: storeItem.name },
            unit_amount: item.quantity * 100,
          },
          quantity: 1,
        };
      }),
      success_url: process.env.CLIENT_SERVER_URL,
      cancel_url: process.env.CLIENT_SERVER_URL,
    });
    res.json({ url: session.url });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

router.get('/donations', async (req, res) => {
  try {
    let charges = [];
    while (true) {
      const result = await stripe.charges.list({ limit: 100 });
      charges.push(...result.data);
      if (!result.has_more) break;
    }
    res.json({ charges });
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
});

router.get('/cumulativeDonations', async (req, res) => {
  let cumulativeTotal = 0;
  try {
    let charges = [];
    while (true) {
      const result = await stripe.charges.list({ limit: 100 });
      charges.push(...result.data);
      if (!result.has_more) break;
    }
    charges.forEach(charge => cumulativeTotal += charge.amount);
    cumulativeTotal = (cumulativeTotal / 100).toFixed(2);
    res.json({ cumulativeTotal });
  } catch (error) {
    res.status(500).json({ error: 'Internal server error' });
  }
});

module.exports = router;
