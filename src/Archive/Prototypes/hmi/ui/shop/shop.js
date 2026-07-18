require('dotenv').config();
const stripe = require('stripe')(process.env.STRIPE_PRIVATE_KEY);

const shopItems = new Map([
    [1, { priceInCents: 10000, name: "Fund Microphone" }],
    [2, { priceInCents: 100, name: "General Donation" }]
])

module.exports = {
    stripe,
    shopItems,

}