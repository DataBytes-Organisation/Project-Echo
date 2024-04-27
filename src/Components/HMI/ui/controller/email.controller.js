const nodemailer = require('nodemailer');
const validation = require('deep-email-validator');
const fs = require('fs');
const path = require('path');

// Function to generate a random 6-digit OTP
function generateOTP() {
    return Math.floor(Math.random() * 900000) + 100000;
}

var transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: 'echodatabytes@gmail.com',
        pass: 'ltzoycrrkpeipngi'
    }
});

async function testEmail(input) {
    let res = await validation.validate(input)
    return {result: res.valid, response: res.validators}
}

exports.send_enquiry = async (email, subject, message) => {
    const validationResult = await testEmail('echodatabytes@gmail.com');
    if (validationResult.result){
        const otp = generateOTP(); // Generate OTP
        
        let mailOptions = {
            from: 'echodatabytes@gmail.com',
            to: `echodatabytes@gmail.com, ${email}`,
            subject: subject,
            text: `Your OTP is: ${otp}\n\n${message}`, // Include OTP in the email message
            attachments: [{
                filename: 'image.png',
                content: fs.createReadStream(path.join(__dirname, '../public/images/tabIcons/logo.png')),
                cid: 'logo@echo.hmi' //same cid value as in the html
            }]
        };

        transporter.sendMail(mailOptions, function (error, info) {
            if (error) {
                console.log(error);
            } else {
                console.log('Email sent: ' + info.response);
                return res.send('<script> alert("user query sent! Please check your mailbox for further communication"); window.location.href = "/"; </script>');
            }
        });
    } else {
        return res.status(400).send("<script> alert(`Sender's email is not valid!`)</script>");
    }
};

exports.verifyOTP = async (req, res) => {
    const { email, otp } = req.body;

    const generatedOTP = generateOTP().toString(); // Generate OTP and convert to string for comparison
    if (otp === generatedOTP) {
        // OTP is correct
        res.status(200).json({ message: 'OTP verified successfully' });
    } else {
        // Incorrect OTP
        res.status(400).json({ error: 'Invalid OTP' });
    }
};
