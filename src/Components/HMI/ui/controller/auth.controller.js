const {client} = require("../middleware")
const axios = require('axios');


const nodemailer = require("nodemailer");

//Guest signup, with random UserId and Password
exports.guestsignup = async (req) => {
  const schema = {
    username: req.username,
    email: req.email,
    password: req.password,
    timestamp: req.timestamp
  };
  try {
    const axiosResponse = await axios.post('http://ts-api-cont:9000/hmi/guestsignup',schema)
  
    if (axiosResponse.status === 201) {
      return { status: 'success' }
    } else {
      console.log("Create user error: ", axiosResponse)
      return { status: "failed", message: "Create Guest Error! Please check the console log" };
    }
  } catch (err) {
    console.log('Status Code: ' + err.response.status + ' ' + err.response.statusText)
    console.log(err.response.data)
    res.status(404).send(`<script> window.location.href = "/login"; alert("Temporary Access Grant exception error occured!");</script>`);
  }

};


exports.signout = async (req, res) => {
  try {
    req.session = null;
    await client.set("JWT", null, (err, res)=> {
      if (err) {
        console.log("Remove JWT Token error: ", err)
      } else {
        console.log("Remove JWT successfully: ")
      }
    })
    await client.set("Roles", null, (err, res)=> {
      if (err) {
        console.log("Remove Roles Token error: ", err)
      } else {
        console.log("Remove Roles successfully: ")
      }
    })
    return res.status(200).send('<script> alert("User logout successfully!"); window.location.href = "/login"</script>')
  } catch (err) {
    this.next(err);
  }
};