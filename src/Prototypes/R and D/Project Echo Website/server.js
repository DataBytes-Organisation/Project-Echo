const express = require('express');
const app = express();
const port = 3000;
const path = require('path');

app.use(express.static(__dirname + "/src/"));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, "/src/index.html"))
})

app.get('/impact', (req, res) => {
    res.sendFile(path.join(__dirname, "/src/impact.html"))
})
app.get('/vision', (req, res) => {
    res.sendFile(path.join(__dirname, "/src/vision.html"))
})

app.listen(port, () => {
    console.log(`Server is running at http://localhost:${port}`);
});