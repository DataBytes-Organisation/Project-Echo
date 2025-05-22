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

    app.get(
        "/api/test/mod",
        [verifyToken, isAdmin], // Assuming 'mod' role is equivalent to 'Admin'
        controller.moderatorBoard
    );

    app.get(
        "/api/test/admin",
        [verifyToken, isAdmin],
        controller.adminBoard
    );

    app.get(
        "/test",
        controller.publicHMI
    );

    app.get(`/user_profile`, async (req, res, next) => {
        let user = await client.get('Users', (err, storedUser) => {
            if (err) {
                return `Error retrieving user role from Redis: ${err}`
            } else {
                return storedUser
            }
        });

        res.send(user);
        next();
    });

  //    app.get('/admin-dashboard', [verifyToken, isAdmin], (req, res) => {
  //     // Access admin-specific data or perform admin actions here
  //     res.send('You have admin access!');
  // });
  
  // app.get('/user', [verifyToken, isLoggedInUser], (req, res) => {
  //     // Access user-specific data or perform actions for logged-in users
  //     res.send('You are a logged-in user!');
  // });
  
  // app.get('/guest', [verifyToken, isGuest], (req, res) => {
  //     // Access guest-specific data or perform actions for guests
  //     res.send('You are a guest!');
  // });
};