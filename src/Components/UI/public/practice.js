const fs = require('fs')

const jsonData = JSON.parse(fs.readFileSync('sample_data.json'));
console.log(jsonData)
