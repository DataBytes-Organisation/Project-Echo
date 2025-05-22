const express = require('express');
const app = express();
const path = require('path');

// Serve static files from the 'ui' directory
app.use(express.static(path.join(__dirname, 'ui')));

app.listen(8080, () => console.log('Server listening on port 8080'));