const { client, authJwt } = require("../middleware");
const controller = require("../controller/user.controller");
const { verifyToken, isAdmin, isLoggedInUser, isGuest } = require('../middleware/authJwt');

module.exports = function(app) {
    app.use(function(req, res, next) {
        res.header(
            "Access-Control-Allow-Headers",
            "Origin, Content-Type, Accept"
        );
        next();
    });

    app.get("/api/test/all", controller.allAccess);

    app.get("/api/test/user", [verifyToken, isLoggedInUser], controller.userBoard);

    app.get("/api/test/mod", [verifyToken, isAdmin], controller.moderatorBoard);

    app.get("/api/test/admin", [verifyToken, isAdmin], controller.adminBoard);

    app.get("/test", controller.publicHMI);

    app.get(`/user_profile`, async (req, res) => {
        try {
            let user = await client.get('Users');
            res.send(user);
        } catch (err) {
            if (!res.headersSent) res.status(500).json({ error: 'Could not load profile' });
        }
    });
};