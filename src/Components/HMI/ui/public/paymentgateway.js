

    function payment(name, Amount, email) {

        var options = {

            "key": "rzp_test_FUo44CLqbVzQj4",
            "amount": Amount,
            "currency": "AUD",
            "name": "Donate Now",
            "description": "Payment Test",
            "image": "https://iconape.com/wp-content/files/sw/12497/png/donate.png",
            "prefill":
            {
                "name": name,
                "email": email,
                "contact": "+61",
            },
            config: {

                display: {
                    blocks: {
                        utib: {
                            name: "Pay using card",
                            instruments: [
                                {
                                    method: "card",
                                },
                            ],
                        }
                    },

                    sequence: ["block.utib"],
                    preferences: {
                        show_default_blocks: false // Should Checkout show its default blocks?
                    }
                }
            },
            "handler": function (response) {
                alert(response.razorpay_payment_id);
            },


            "modal": {
                "ondismiss": function () {
                    paymentcancel();
                }
            },
            "handler":
                function (response) {
                    if (response.razorpay_payment_id) {
                        //payment sucessfull
                    }

                }
        };
        return options;
    }

   
    function paymentcancel() {
        const responseDiv = document.getElementById('payment-response');
        const processingIndicator = document.getElementById('processing-indicator');
        const crossmark = document.querySelector('.crossmark');
        const buttonContainer = document.querySelector('.button-container');
        const retryButton = document.querySelector('.retry-button');
        const overlay = document.querySelector('.overlay1');
       
        // Duration of the processing indicator animation
        const processingDuration = 2000; // Adjust to match your actual duration
    
        responseDiv.classList.remove('hidden');
        processingIndicator.classList.remove('hidden');
    
        // Show processing indicator and hide crossmark initially
        processingIndicator.style.display = 'block';
        overlay.style.display = 'flex';    
        crossmark.style.display = 'none';
    
        // After the processing duration, start crossmark animation
        setTimeout(() => {
            processingIndicator.style.display = 'none';
            crossmark.style.display = 'flex'; // Make it visible
            crossmark.style.opacity = '1'; // Fade in
    
            // Show the button container after the crossmark animation
           
            setTimeout(() => {
                buttonContainer.style.display = 'flex';
                retryButton.classList.add('shake-animation');
            }, 800);
    
        }, processingDuration);
    }