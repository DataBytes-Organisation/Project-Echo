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


exports.signout = async (req, res, next) => {
  try {
    await client.del("JWT", (err, res)=> {
      if (err) {
        console.log("Remove JWT Token error: ", err)
      } else {
        console.log("Remove JWT successfully: ")
      }
    })
    await client.del("Roles", (err, res)=> {
      if (err) {
        console.log("Remove Roles Token error: ", err)
      } else {
        console.log("Remove Roles successfully: ")
      }
    })
    return res.status(200).send('<script> alert("User logout successfully!"); window.location.href = "/login"</script>')
  } catch (err) {
    next(err);
  }
};

//Delete Account functionality
// exports.deleteaccount = async (req, res) => { 
//     try {
//         const jwtToken = req.session.jwtToken;
        
//         const axiosResponse = await axios.delete('http://ts-api-cont:9000/hmi/delete-account', {
//             headers: {
//                 Authorization: `Bearer ${jwtToken}`
//             }
//         });

//         if (axiosResponse.status === 200) {
//           return res.status(200).send('<script> alert("User account deleted successfully!"); window.location.href = "/login"</script>')
//         } else {
//           return res.status(500).send('<script> alert("Failed to delete user account!");');
//         }
//       } catch (err) {
//         console.error("Delete Account Error:", err);
//         return res.status(500).send('<script> alert("An error occured while deleting the user account!");');
//     }
// };
