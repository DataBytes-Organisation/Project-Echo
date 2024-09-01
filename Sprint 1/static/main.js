// Event listener for the upload button
document.getElementById('uploadButton').addEventListener('click', function() {
    // Get the selected image and audio files
    let imageFile = document.getElementById('imageInput').files[0];
    let audioFile = document.getElementById('audioInput').files[0];

    // Clear previous media and tags displays
    document.getElementById('mediaDisplay').innerHTML = '';
    document.getElementById('tagsDisplay').innerHTML = '';

    // Check if an image file is selected
    if (!imageFile) {
        alert("Please select an image to upload.");
        return;
    }

    // Create a FormData object to hold the files
    let formData = new FormData();
    formData.append('image', imageFile);
    if (audioFile) {
        formData.append('audio', audioFile);
    }

    // Send the files to the backend using Fetch API
    fetch('http://127.0.0.1:5000/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.error('Error:', error);
    });
    
    .then(response => response.json())
    .then(data => {
        // Display the tags returned by the backend
        let tagsDisplay = document.getElementById('tagsDisplay');
        tagsDisplay.innerHTML = 'Tags: ' + data.tags.join(', ');

        // Display the uploaded image
        let mediaDisplay = document.getElementById('mediaDisplay');
        let img = document.createElement('img');
        img.src = URL.createObjectURL(imageFile);
        img.alt = "Uploaded Image";
        img.style.maxWidth = "100%";
        mediaDisplay.appendChild(img);

        // If audio was uploaded, display the audio player
        if (audioFile) {
            let audio = document.createElement('audio');
            audio.controls = true;
            audio.src = URL.createObjectURL(audioFile);
            mediaDisplay.appendChild(audio);
        }

        // Add the image to the gallery
        let gallery = document.getElementById('animalGallery');
        let newImage = document.createElement('img');
        newImage.src = URL.createObjectURL(imageFile);
        newImage.classList.add('col-md-3');
        gallery.appendChild(newImage);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
