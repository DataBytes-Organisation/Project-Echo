const jwt = require('jsonwebtoken');
const config = require('../config/auth.config');

function verifyToken(token) {
    try {
        return jwt.verify(token, config.secret);
    } catch (error) {
        throw new Error('Invalid token');
    }
}

function getUserIdFromToken(token) {
    try {
        const decoded = verifyToken(token);
        return decoded.id;
    } catch (error) {
        return null;
    }
}

module.exports = {
    verifyToken,
    getUserIdFromToken
};