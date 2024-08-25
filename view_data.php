<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <title>View Uploaded Data</title>
</head>
<body>
    <div class="container mt-5">
        <h2>Uploaded Animal Data</h2>
        <table class="table table-bordered">
            <thead>
                <tr>
                    <th>Animal Name</th>
                    <th>Animal Type</th>
                    <th>Conservation Status</th>
                    <th>Audio</th>
                    <th>Image</th>
                </tr>
            </thead>
            <tbody>
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

                $sql = "SELECT animal_name, animal_type, conservation_status, audio_path, image_path FROM animals";
                $result = $conn->query($sql);

                if ($result->num_rows > 0) {
                    while($row = $result->fetch_assoc()) {
                        echo "<tr>";
                        echo "<td>" . htmlspecialchars($row["animal_name"]) . "</td>";
                        echo "<td>" . htmlspecialchars($row["animal_type"]) . "</td>";
                        echo "<td>" . htmlspecialchars($row["conservation_status"]) . "</td>";
                        echo "<td><audio controls><source src='" . htmlspecialchars($row["audio_path"]) . "' type='audio/mpeg'></audio></td>";
                        echo "<td><img src='" . htmlspecialchars($row["image_path"]) . "' style='width:100px;height:100px;'></td>";
                        echo "</tr>";
                    }
                } else {
                    echo "<tr><td colspan='5'>No data found</td></tr>";
                }

                $conn->close();
                ?>
            </tbody>
        </table>
    </div>

    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.0.6/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
