<?php
// Get the raw POST data
$data = json_decode(file_get_contents("php://input"), true);

$animalId = $data['animalId'];
$section = $data['section'];
$newStatus = $data['status'];

// Load existing requests
$filename = "requests.json";
$requests = json_decode(file_get_contents($filename), true);
$updated = false;

// Update matching request
foreach ($requests as &$req) {
  if ($req['animalId'] === $animalId && $req['section'] === $section && $req['status'] === "Pending Review") {
    $req['status'] = $newStatus;
    $req['timestamp'] = date("Y-m-d H:i:s");
    $updated = true;
    break;
  }
}

// Save if updated
if ($updated) {
  file_put_contents($filename, json_encode($requests, JSON_PRETTY_PRINT));
  echo "✅ Status updated to: $newStatus";
} else {
  echo "⚠️ Request not found or already reviewed.";
}
?>
