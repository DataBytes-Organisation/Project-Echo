// only used by Jest to transform the browser-side ES modules (import/export)
// during test runs - the actual app still serves these as plain <script type="module">
// files to the browser, this doesn't touch that at all
module.exports = {
  presets: [['@babel/preset-env', { targets: { node: 'current' } }]],
};
