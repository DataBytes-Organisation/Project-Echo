const jwt = require('jsonwebtoken');
const config = require('../config/auth.config');
const User = require('../model/user.model');
// Middleware to verify the token and extract user role
const verifyToken = (req, res, next) => {
    let token = req.session.token
    
    if (!token) {
        return res.status(403).send({ message: "No token provided!" });
    }

    jwt.verify(token, config.secret, (err, decoded) => {
        if (err) {
            return res.status(401).send({ message: "Unauthorized!" });
        }
        req.userId = decoded.id;
        next();
    });
};

// Middleware to check if the user is an Admin
const isAdmin = async (req, res, next) => {
    try {
        const user = await User.findById(req.userId);
        if (user.role === 'Admin') {
            next();
        } else {
            res.status(403).send({ message: "Require Admin Role!" });
        }
    } catch (err) {
        res.status(500).send({ message: "Unable to validate user role." });
    }
};

// Middleware to check if the user is a Logged in user
const isLoggedInUser = async (req, res, next) => {
    try {
        const user = await User.findById(req.userId);
        if (user.role === 'Logged in user' || user.role === 'Admin') { 
            next();
        } else {
            res.status(403).send({ message: "Require Logged in user Role!" });
        }
    } catch (err) {
        res.status(500).send({ message: "Unable to validate user role." });
    }
};

// Middleware to check if the user is a Guest
const isGuest = async (req, res, next) => {
    try {
        const user = await User.findById(req.userId);
        if (user.role === 'Guest' || user.role === 'Logged in user' || user.role === 'Admin') {
            next();
        } else {
            res.status(403).send({ message: "Require Guest Role!" });
        }
    } catch (err) {
        res.status(500).send({ message: "Unable to validate user role." });
    }
};

module.exports = {
    verifyToken,
    isAdmin,
    isLoggedInUser,
    isGuest
};
