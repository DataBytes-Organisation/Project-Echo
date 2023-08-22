const path = require("path");

module.exports = {
  entry: [
    "./src/js/animation.js",
    "./src/js/audio_recorder.js",
    "./src/js/bootstrap.bundle.min.js",
    "./src/js/HMI-utils.js",
    "./src/js/HMI.js",
    "./src/js/ol.js",
  ],
  output: {
    filename: "bundle.js",
    path: path.resolve(__dirname, "dist"),
  },
};
