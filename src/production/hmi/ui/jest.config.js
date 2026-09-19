module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/__tests__/**/*.test.js'],
  // don't want jest trying to pick up its own installed copy inside node_modules
  testPathIgnorePatterns: ['/node_modules/'],
};
