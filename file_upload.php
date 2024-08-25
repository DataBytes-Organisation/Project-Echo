<?php
$servername = "localhost";
$username = "root";
$password = ""; // Add your database password if any
$dbname = "animal_data_db";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Handle the file upload
$targetDir = "uploads/";
$audioFileName = basename($_FILES["animalAudio"]["name"]);
$imageFileName = basename($_FILES["animalImage"]["name"]);

$audioTargetFilePath = $targetDir . $audioFileName;
$imageTargetFilePath = $targetDir . $imageFileName;

$uploadOk = 1;

// Check if image file is a valid image or not
if(isset($_FILES["animalImage"]["tmp_name"])) {
    $check = getimagesize($_FILES["animalImage"]["tmp_name"]);
    if($check !== false) {
        $uploadOk = 1;
    } else {
        echo "File is not an image.";
        $uploadOk = 0;
    }
}

// Check if $uploadOk is set to 0 by an error
if ($uploadOk == 0) {
    echo "Sorry, your file was not uploaded.";
// If everything is ok, try to upload file
} else {
    if (move_uploaded_file($_FILES["animalAudio"]["tmp_name"], $audioTargetFilePath) && 
        move_uploaded_file($_FILES["animalImage"]["tmp_name"], $imageTargetFilePath)) {
        
        // Prepare the SQL statement
        $stmt = $conn->prepare("INSERT INTO animals (animal_name, animal_type, conservation_status, audio_path, image_path) VALUES (?, ?, ?, ?, ?)");
        $stmt->bind_param("sssss", $animalName, $animalType, $conservationStatus, $audioTargetFilePath, $imageTargetFilePath);

        // Get the form data
        $animalName = $_POST['animalName'];
        $animalType = $_POST['animalType'];
        $conservationStatus = $_POST['conservationStatus'];

        // Execute the SQL statement
        if ($stmt->execute()) {
            echo "The files have been uploaded successfully.";
        } else {
            echo "Error: " . $stmt->error;
        }
        
        $stmt->close();
    } else {
        echo "Sorry, there was an error uploading your files.";
    }
}

$conn->close();
?>
