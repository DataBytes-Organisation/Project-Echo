<?php


if($_POST["query"]) {
$to = "viettung0901@gmail.com";
$from = $_POST["email"];
$subject = "New Query For Project Echo!";

$email = "
    <html>
    <head>
    <title>Project Echo Query</title>
    </head>
    <body>
        <h1>You receive a query!</h1>
        <p>" . $_POST["query"] . "</p>
    </body>
    </html>
";


// Always set content-type when sending HTML email
$headers = "MIME-Version: 1.0" . "\r\n";
$headers .= "Content-type:text/html;charset=UTF-8" . "\r\n";

// More headers
$headers .= 'From: <'. $from . ">\r\n";
$headers .= 'Cc: nguyenviet@deakin.edu.au' . "\r\n";


mail($to,$subject,$message,$headers);


}


?>

