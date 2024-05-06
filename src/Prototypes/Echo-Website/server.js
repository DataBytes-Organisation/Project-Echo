const express = require('express');
const app = express();
const port = 3000;

app.use(express.static(__dirname + "/main/"));

app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});