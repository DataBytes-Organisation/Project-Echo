require("dotenv").config();
const fs = require("fs");
const https = require("https");
const http = require("http");
const express = require("express");
const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const { createProxyMiddleware } = require("http-proxy-middleware");
const cookieParser = require("cookie-parser");
const session = require("express-session");

const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
});

const projectSchema = new mongoose.Schema({
  name: { type: String, required: true, unique: true },
  owner: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  acceptedUsers: [{ type: mongoose.Schema.Types.ObjectId, ref: "User" }],
  accessRequests: [{ type: mongoose.Schema.Types.ObjectId, ref: "User" }],
  labels: { type: Object, default: {} }, // Add labels field
});

projectSchema.pre("save", function (next) {
  this.acceptedUsers = [
    ...new Set(this.acceptedUsers.map((user) => user.toString())),
  ];
  next();
});

const User = mongoose.model("User", userSchema);
const Project = mongoose.model("Project", projectSchema);

class ObjServer {
  constructor() {
    this.app = express();
    this.port = process.env.PORT || 3123;
    this.nextJsPort = process.env.NEXTJS_PORT || 3000;
    this.proxyTo = process.env.PROXY_TARGET || "host.docker.internal";
    this.useHttps = process.env.USE_HTTPS === "true";

    if (this.useHttps) {
      this.sslOptions = {
        key: fs.readFileSync(process.env.SSL_KEY_PATH),
        cert: fs.readFileSync(process.env.SSL_CERT_PATH),
        agent: new https.Agent({ rejectUnauthorized: false }), // Disable SSL certificate validation
      };
    }

    this.JWT_SECRET = process.env.JWT_SECRET || "your_jwt_secret";

    this.connectDatabase();
    this.initializeMiddlewares();
    this.initializeRoutes();
  }

  async connectDatabase() {
    try {
      await mongoose.connect(process.env.MONGO_URI);
      console.log("MongoDB connected");

      // Switch to the specified database
      this.db = mongoose.connection.useDb(process.env.MONGODB);
      console.log(`Switched to database: ${process.env.MONGODB}`);

      // Create the models using the new database connection
      this.User = this.db.model("User", userSchema);
      this.Project = this.db.model("Project", projectSchema);
    } catch (err) {
      console.error("MongoDB connection error:", err);
    }
  }

  initializeMiddlewares() {
    this.app.use(express.json());
    this.app.use(cookieParser());
    this.app.use(express.urlencoded({ extended: true }));
    this.app.use(
      session({
        secret: process.env.SESSION_SECRET || "your_session_secret",
        resave: false,
        saveUninitialized: true,
      })
    );
  }

  initializeRoutes() {
    this.app.post("/api/register", this.registerUser.bind(this));
    this.app.post("/api/login", this.loginUser.bind(this));
    this.app.post("/api/logout", this.logoutUser.bind(this));
    this.app.get(
      "/api/check-auth",
      this.verifyToken.bind(this),
      this.checkAuth.bind(this)
    );
    this.app.get(
      "/api/profile",
      this.verifyToken.bind(this),
      this.getProfile.bind(this)
    );
    this.app.post(
      "/api/projects",
      this.verifyToken.bind(this),
      this.createProject.bind(this)
    );
    this.app.get(
      "/api/projects",
      this.verifyToken.bind(this),
      this.getProjects.bind(this)
    );

    this.app.post(
      "/api/projects/request-access/:projectId",
      this.verifyToken.bind(this),
      this.requestAccessToProject.bind(this)
    );
    this.app.get(
      "/api/projects/access-requests/:projectId",
      this.verifyToken.bind(this),
      this.getProjectAccessRequests.bind(this)
    );
    this.app.post(
      "/api/projects/accept-request",
      this.verifyToken.bind(this),
      this.acceptProjectAccessRequest.bind(this)
    );

    this.app.get(
      "/api/projects/:projectId/labels",
      this.verifyToken.bind(this),
      this.getProjectLabels.bind(this)
    );
    this.app.post(
      "/api/projects/:projectId/labels",
      this.verifyToken.bind(this),
      this.updateProjectLabels.bind(this)
    );

    const noCacheMiddleware = (req, res, next) => {
      res.setHeader(
        "Cache-Control",
        "no-store, no-cache, must-revalidate, proxy-revalidate"
      );
      res.setHeader("Pragma", "no-cache");
      res.setHeader("Expires", "0");
      res.setHeader("Surrogate-Control", "no-store");
      next();
    };

    this.app.use(noCacheMiddleware);

    this.app.use(
      "/pback",
      createProxyMiddleware({
        target: `http://${this.proxyTo}:50999`,
        changeOrigin: true,
        pathRewrite: {
          "^/pback": "",
        },
        timeout: 30000, // Increase timeout to 30 seconds
        proxyTimeout: 30000, // Increase proxy timeout to 30 seconds
        onProxyRes: function (proxyRes, req, res) {
          // Disable caching on the proxy response
          proxyRes.headers["Cache-Control"] =
            "no-store, no-cache, must-revalidate, proxy-revalidate";
          proxyRes.headers["Pragma"] = "no-cache";
          proxyRes.headers["Expires"] = "0";
          proxyRes.headers["Surrogate-Control"] = "no-store";
        },
      })
    );

    this.app.use(
      "/mainmap",
      createProxyMiddleware({
        target: `http://${this.proxyTo}:8080/map`,
        changeOrigin: true,
        pathRewrite: {
          "^/map": "",
        },
        timeout: 30000, // Increase timeout to 30 seconds
        proxyTimeout: 30000, // Increase proxy timeout to 30 seconds
        onProxyRes: function (proxyRes, req, res) {
          // Disable caching on the proxy response
          proxyRes.headers["Cache-Control"] =
            "no-store, no-cache, must-revalidate, proxy-revalidate";
          proxyRes.headers["Pragma"] = "no-cache";
          proxyRes.headers["Expires"] = "0";
          proxyRes.headers["Surrogate-Control"] = "no-store";
        },
      })
    );

    
    const folders = ["js", "images", "css", "audio","md_files"]; // Add your folders here

folders.forEach((folder) => {
  this.app.use(
    `/${folder}`,
    createProxyMiddleware({
      target: `http://${this.proxyTo}:8080/${folder}`,
      changeOrigin: true,
      pathRewrite: {
        [`^/${folder}`]: "",
      },
      timeout: 30000, // Increase timeout to 30 seconds
      proxyTimeout: 30000, // Increase proxy timeout to 30 seconds
      onProxyRes: function (proxyRes, req, res) {
        // Disable caching on the proxy response
        proxyRes.headers["Cache-Control"] =
          "no-store, no-cache, must-revalidate, proxy-revalidate";
        proxyRes.headers["Pragma"] = "no-cache";
        proxyRes.headers["Expires"] = "0";
        proxyRes.headers["Surrogate-Control"] = "no-store";
      },
    })
  );
});

// Proxy additional routes
const routes = [
  { path: "/movement_time/:start/:end", target: "/movement_time" },
  { path: "/events_time/:start/:end", target: "/events_time" },
  { path: "/microphones", target: "/microphones" },
  { path: "/audio/:id", target: "/audio" },
  { path: "/post_recording", target: "/post_recording", method: "POST" },
  { path: "/sim_control/:control", target: "/sim_control", method: "POST" },
  { path: "/latest_movement", target: "/latest_movement" }
];

routes.forEach((route) => {
  this.app[route.method?.toLowerCase() || "get"](
    route.path,
    createProxyMiddleware({
      target: `http://${this.proxyTo}:8080${route.target}`,
      changeOrigin: true,
      pathRewrite: {
        [`^${route.path.split('/:')[0]}`]: "",
      },
      timeout: 30000, // Increase timeout to 30 seconds
      proxyTimeout: 30000, // Increase proxy timeout to 30 seconds
      onProxyRes: function (proxyRes, req, res) {
        // Disable caching on the proxy response
        proxyRes.headers["Cache-Control"] =
          "no-store, no-cache, must-revalidate, proxy-revalidate";
        proxyRes.headers["Pragma"] = "no-cache";
        proxyRes.headers["Expires"] = "0";
        proxyRes.headers["Surrogate-Control"] = "no-store";
      },
    })
  );
});


    /*this.app.use(
      "/phealthcheck",
      this.verifyToken.bind(this),
      createProxyMiddleware({
        target: `http://${this.proxyTo}:50999`, // Replace with your Python backend URL
        changeOrigin: true,
        pathRewrite: {
          "^/phealthcheck": "/phealthcheck",
        },
      })
    );*/

    this.app.use(
      "/",
      createProxyMiddleware({
        target: `http://${this.proxyTo}:${this.nextJsPort}`,
        changeOrigin: true,
        pathRewrite: {
          "^/": "/", // remove base path
        },
      })
    );
  }

  async requestAccessToProject(req, res) {
    const { projectId } = req.params;
    const user = await this.User.findOne({ username: req.username });
    try {
      const project = await this.Project.findById(projectId);
      if (project) {
        project.accessRequests.push(user._id);
        await project.save();
        res.status(200).json({ message: "Access request sent" });
      } else {
        res.status(404).json({ error: "Project not found" });
      }
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async getProjectAccessRequests(req, res) {
    const { projectId } = req.params;
    const project = await this.Project.findById(projectId).populate(
      "accessRequests"
    );
    res.status(200).json({ accessRequests: project.accessRequests });
  }

  async getProjects(req, res) {
    const user = await this.User.findOne({ username: req.username });
    const projects = await this.Project.find().populate("owner acceptedUsers");

    // Filter projects to include only accepted details if the user is on the accepted list
    const filteredProjects = projects.map((project) => {
      const isAccepted = project.acceptedUsers.some((acceptedUser) =>
        acceptedUser._id.equals(user._id)
      );
      return {
        _id: project._id,
        name: project.name,
        owner: project.owner.username,
        isAccepted,
      };
    });

    res.status(200).json({ projects: filteredProjects });
  }

  async acceptProjectAccessRequest(req, res) {
    const { projectId, userId } = req.body;
    try {
      const project = await this.Project.findById(projectId);
      if (project) {
        // Check if the user is already in the acceptedUsers array
        if (!project.acceptedUsers.includes(userId)) {
          project.acceptedUsers.push(userId);
        }
        // Remove the user from the accessRequests array
        project.accessRequests = project.accessRequests.filter(
          (requestId) => !requestId.equals(userId)
        );
        await project.save();
        res.status(200).json({ message: "User accepted to the project" });
      } else {
        res.status(404).json({ error: "Project not found" });
      }
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async getProjectLabels(req, res) {
    const { projectId } = req.params;
    try {
      const project = await this.Project.findById(projectId);
      const user = await this.User.findOne({ username: req.username });
      if (project) {
        const isAccepted = project.acceptedUsers.includes(user._id);
        if (isAccepted) {
          res.status(200).json({ labels: project.labels });
        } else {
          res
            .status(403)
            .json({ error: "You do not have access to this project" });
        }
      } else {
        res.status(404).json({ error: "Project not found" });
      }
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async updateProjectLabels(req, res) {
    const { projectId } = req.params;
    const { labels } = req.body;
    try {
      const project = await this.Project.findById(projectId);
      const user = await this.User.findOne({ username: req.username });
      if (project) {
        const isAccepted = project.acceptedUsers.includes(user._id);
        if (isAccepted) {
          project.labels = labels;
          await project.save();
          res.status(200).json({ message: "Labels updated successfully" });
        } else {
          res
            .status(403)
            .json({ error: "You do not have access to this project" });
        }
      } else {
        res.status(404).json({ error: "Project not found" });
      }
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async registerUser(req, res) {
    const { username, password, email } = req.body;
    try {
      const hashedPassword = await bcrypt.hash(password, 10);
      const user = new this.User({ username, email, password: hashedPassword });
      await user.save();
      res.status(200).json({ message: "User registered successfully" });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async loginUser(req, res) {
    const { username, password } = req.body;
    try {
      const user = await this.User.findOne({ username });
      if (!user) {
        return res.status(400).json({ error: "Invalid username or password" });
      }
      const isPasswordValid = await bcrypt.compare(password, user.password);
      if (!isPasswordValid) {
        return res.status(400).json({ error: "Invalid username or password" });
      }
      const token = jwt.sign({ username }, this.JWT_SECRET, {
        expiresIn: "1h",
      });
      res.cookie("token", token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        maxAge: 3600000, // 1 hour
        sameSite: "strict",
        path: "/",
      });
      res.status(200).json({ message: "Logged in successfully" });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async logoutUser(req, res) {
    res.clearCookie("token");
    res.status(200).json({ message: "Logged out successfully" });
  }

  verifyToken(req, res, next) {
    const token = req.cookies.token || req.headers.authorization?.split(" ")[1];
    if (!token) {
      return res.status(403).json({ error: "No token provided" });
    }
    jwt.verify(token, this.JWT_SECRET, (err, decoded) => {
      if (err) {
        return res.status(401).json({ error: "Failed to authenticate token" });
      }
      req.username = decoded.username;
      next();
    });
  }

  checkAuth(req, res) {
    res.status(200).json({ username: req.username });
  }

  getProfile(req, res) {
    res.status(200).json({ message: `Welcome ${req.username}` });
  }

  async createProject(req, res) {
    const { name } = req.body;
    const owner = await this.User.findOne({ username: req.username });
    const project = new this.Project({
      name,
      owner: owner._id,
      acceptedUsers: [owner._id],
    });
    try {
      await project.save();
      res
        .status(200)
        .json({ message: "Project created successfully", project });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }

  async getUserProjects(req, res) {
    const user = await this.User.findOne({ username: req.username });
    const projects = await this.Project.find({ acceptedUsers: user._id });
    res.status(200).json({ projects });
  }

  start() {
    if (this.useHttps) {
      https.createServer(this.sslOptions, this.app).listen(this.port, () => {
        console.log(`HTTPS Server running on port ${this.port}`);
      });
      this.startHttpRedirect();
    } else {
      http.createServer(this.app).listen(this.port, () => {
        console.log(`HTTP Server running on port ${this.port}`);
      });
    }
  }

  startHttpRedirect() {
    http
      .createServer((req, res) => {
        res.writeHead(301, {
          Location: `https://${req.headers.host}${req.url}`,
        });
        res.end();
      })
      .listen(80, () => {
        console.log("HTTP server listening on port 80, redirecting to HTTPS");
      });
  }
}

module.exports = ObjServer;
