//
// Audio data example
// Minosha Gamage
//
;
var data = []; // start with empty array
var position = -1; // current position 
function doAdd() {
    document.getElementById("message").innerHTML = ""; //clear message
    var idString = document.getElementById("AudioID").value;
    var id = parseInt(idString);
    if (isNaN(id)) {
        document.getElementById("message").innerHTML =
            "Audio ID must be a number!"; // error message
        return; // do nothing
    }
    var species = document.getElementById("Species").value;
    if (species == "") {
        document.getElementById("message").innerHTML =
            "Species must be entered";
        return; // do nothing
    }
	var recording = document.getElementById("Recording").value;
    if (recording == "") {
        document.getElementById("message").innerHTML =
            "Recording must be entered";
        return; // do nothing
    }
	var timestamp = document.getElementById("Timestamp").value;
    if (timestamp == "") {
        document.getElementById("message").innerHTML =
            "Timestamp must be entered";
        return; // do nothing
    }
	var latitude = document.getElementById("Latitude").value;
    if (latitude == "") {
        document.getElementById("message").innerHTML =
            "Latitude must be entered";
        return; // do nothing
    }
	var longitude = document.getElementById("Longitude").value;
    if (longitude == "") {
        document.getElementById("message").innerHTML =
            "Longitude must be entered";
        return; // do nothing
    }
    // we can add an element to end of data
    position = data.length; // one past end of array
    data[position] = { ID: id, Species: species, Recording: recording, Timestamp: timestamp, Latitude: latitude, Longitude: longitude };
    updateDisplay();
}
function doNext() {
    if (position >= data.length - 1) {
        document.getElementById("message").innerHTML =
            "Already at end of list"; // error message
        return; // do nothing
    }
    position++; // back up position
    updateDisplay();
}
function doPrev() {
    if (position == 0) {
        document.getElementById("message").innerHTML =
            "Already at start of list"; // error message
        return; // do nothing
    }
    position--; // back up position
    updateDisplay();
}
function doDelete() {
    if (position < 0) {
        document.getElementById("message").innerHTML =
            "List is empty"; // error message
        return; // do nothing
    }
    // delete array element at current position
    data.splice(position, 1); // tricky javascript code
    if (data.length == position)
        position--; // deleted last element in list
    updateDisplay();
}
function updateDisplay() {
    var posString = "";
    if (position >= 0) {
        var current = data[position];
        document.getElementById("AudioID").value = "" + current.ID;
        document.getElementById("Species").value = current.Species;
		document.getElementById("Recording").value = current.Recording;
		document.getElementById("Timestamp").value = "" + current.Timestamp;
        document.getElementById("Latitude").value = current.Latitude;
		document.getElementById("Longitude").value = current.Longitude;
        posString = "record " + (position + 1) + " of " + data.length;
    }
    else {
        document.getElementById("AudioID").value = "";
        document.getElementById("Species").value = "";
		document.getElementById("Recording").value = "";
		document.getElementById("Timestamp").value = "";
        document.getElementById("Latitude").value = "";
		document.getElementById("Longitude").value = "";
        posString = "Audio list is empty";
    }
    document.getElementById("currentRecord").innerHTML = posString;
}
