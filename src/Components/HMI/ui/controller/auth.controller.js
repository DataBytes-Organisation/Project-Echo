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
exports.deleteaccount = async (req, res) => { 
    try {
        const jwtToken = req.session.jwtToken;
        
        const axiosResponse = await axios.delete('http://ts-api-cont:9000/hmi/delete-account', {
            headers: {
                Authorization: `Bearer ${jwtToken}`
            }
        });

        if (axiosResponse.status === 200) {
          return res.status(200).send('<script> alert("User account deleted successfully!"); window.location.href = "/login"</script>')
        } else {
          return res.status(500).send('<script> alert("Failed to delete user account!");');
        }
      } catch (err) {
        console.error("Delete Account Error:", err);
        return res.status(500).send('<script> alert("An error occured while deleting the user account!");');
    }
};

//Change the Password
exports.changepassword = async (req, res) => {
  let oldpw = req.body.oldPassword;
  let newpd = req.body.newPassword;
  let cfm_newpw = req.body.confirmPassword
    
  try {
    const jwtToken = req.session.jwtToken;
    const axiosResponse = await axios.post('http://ts-api-cont:9000/hmi/ChangePassword', {
      headers: {
        Authorization: `Bearer ${jwtToken}`
      },
      oldpw: oldpw,
      newpd: newpd,
      cfm_newpw: cfm_newpw
    });
    
    if (axiosResponse.status === 200) {
      console.log('Status Code: ' + axiosResponse.status + ' ' + axiosResponse.statusText)
      console.log("Change Password response: ", axiosResponse.data);
      
      
      res.status(200).send(
      `<script> 
        alert("Change Password Successfully");
        window.location.href = "/welcome"
      </script>`);
        
      
    } else {
      console.log("Login response: ", axiosResponse.data);
      res.status(400).send('<script> window.location.href = "/login"; alert("Failed! Invalid credentials!");</script>');
    }
  } catch (err) {
    console.log('Login exception error: ' + err)
    res.send(`<script> alert("Change Password exception Error: ${err}!");</script>`);
  }  
};
