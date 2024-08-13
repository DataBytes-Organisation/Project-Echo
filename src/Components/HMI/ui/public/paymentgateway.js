

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
                    if (confirm("Are you sure, you want to close the form?")) {
                        txt = "You pressed OK!";
                        console.log("Checkout form closed by the user");
                    } else {
                        txt = "You pressed Cancel!";
                        console.log("Complete the Payment")
                    }
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

   
