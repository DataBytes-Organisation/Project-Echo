const ObjServer = require("./ObjServer");

(async () => {
  console.log("Welcome to APISessionManager");
  const server = new ObjServer();
  server.start();
})();
