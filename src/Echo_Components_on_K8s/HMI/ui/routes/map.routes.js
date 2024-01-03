const { verifySignUp, client } = require("../middleware");
const axios = require('axios');


// const MESSAGE_API_URL = 'http://localhost:9000/hmi';
const MESSAGE_API_URL = 'http://ts-api-cont:9000/hmi';
let token;
// client.get('JWT', (err, storedToken) => {
//   if (err) {
//     console.error('Error retrieving token from Redis:', err);
//     return null
//   } else {
//     console.log('Stored Token:', storedToken);
//     return storedToken
//   }
// }).then(response => token = response)

module.exports = function(app) {
  

  app.use(function(req, res, next) {
    // req.headers['Authorization'] = `Bearer ${token}`;
    res.header(
      "Access-Control-Allow-Headers",
      "Origin, Content-Type, Accept"
    );
    // console.log("Token inside map requests: ", token);
    next();
  });

  app.get(`/movement_time/:start/:end`, async (req, res, next) => {
    const start = req.params.start
    const end = req.params.end
    const response = await axios.get(`${MESSAGE_API_URL}/movement_time?start=${start}&end=${end}`);
    if (Object.keys(response.data).length === 0) {
      res.send([])
    } else {
      res.send(response.data);
    }
  })

  app.get(`/events_time/:start/:end`, async (req, res, next) => {
    const start = req.params.start
    const end = req.params.end
    const response = await axios.get(`${MESSAGE_API_URL}/events_time?start=${start}&end=${end}`);
    res.send(response.data);
    next()
  })

  app.get(`/microphones`, async (req, res, next) => {
    const response = await axios.get(`${MESSAGE_API_URL}/microphones`);
    res.send(response.data);
    next()
  })

  app.get(`/audio/:id`, async (req, res, next) => {
    const id = req.params.id;
    //console.log(`${MESSAGE_API_URL}/audio?id=${id}`);
    const response = await axios.get(`${MESSAGE_API_URL}/audio?id=${id}`);
    res.send(response.data);
    next()
  })

  app.post(`/post_recording`, async (req, res, next) => {
    let data = req.body
    axios.post(`${MESSAGE_API_URL}/post_recording`, data)
    .then(response => {
      console.log('Record response:', response.data);
    })
    .catch(error => {
      console.error('Error:', error);
    });
    //axios.post(`${MESSAGE_API_URL}/post_recording?data=${recordingData}`);
    next()
  })

  app.post(`/sim_control/:control`, async (req, res, next) => {
    const control = req.params.control
    const response = await axios.post(`${MESSAGE_API_URL}/sim_control?control=${control}`);
    res.send(response.data);
    next()
  })

  
  app.get(`/latest_movement`, async (req, res, next) => {
    const response = await axios.get(`${MESSAGE_API_URL}/latest_movement`);
    res.send(response.data);
    next()
  })
}