

    function payment(name, order, email, onSettled = function () {}) {

        var options = {

            "key": order.keyId,
            "amount": order.amount,
            "currency": order.currency,
            "order_id": order.orderId,
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
            //"handler": function (response) {
            //    alert(response.razorpay_payment_id);
            //},


            "modal": {
                "ondismiss": function () {
                    paymentcancel();
                    onSettled();
                }
            },
            "handler": async function (response) {
                try {
                    var verificationResponse = await fetch('/api/save-razorpay-payment', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            paymentId: response.razorpay_payment_id,
                            orderId: response.razorpay_order_id,
                            signature: response.razorpay_signature,
                            name: options.prefill.name
                        })
                    });
                    if (!verificationResponse.ok) throw new Error('Payment verification failed');
                    await verificationResponse.json();
                    if (typeof window !== 'undefined' && window.showToast) {
                        window.showToast('Donation verified. Thank you!', 'success');
                    }
                } catch (error) {
                    if (typeof window !== 'undefined' && window.showToast) {
                        window.showToast('Payment could not be verified. Please contact support.', 'error');
                    }
                } finally {
                    onSettled();
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

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = { payment };
    }
