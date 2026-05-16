confirmPassword = (req, res, next) => {
  if (req.body.confirmpassword != req.body.password) {
    // res.status(400).send({
    //   message: `Failed! Password does not match its confirmation!`
    // });
    res.status(400).send('<script> window.location.href = "/login"; alert("Failed! Password does not match its confirmation!");</script>');
    return;
  }
  next();
}

const verifySignUp = {
  confirmPassword
};

module.exports = verifySignUp;