<?php
$animalName = $_POST['animalId'];
$discription = $_POST['suggestion'];

// Connect to MySQL database
$conn = new mysqli('localhost', 'root', '', 'animal_info');

// Check connection
if ($conn->connect_error) {
    die('Connection Failed: ' . $conn->connect_error);
} else {
    $stmt = $conn->prepare("INSERT INTO request (animalId, suggestion) VALUES (?, ?)");
    $stmt->bind_param("ss", $animalName, $discription);

    if ($stmt->execute()) {
        echo "success";
    } else {
        echo "error: " . $stmt->error;
    }

    $stmt->close();
    $conn->close();
}
?>
