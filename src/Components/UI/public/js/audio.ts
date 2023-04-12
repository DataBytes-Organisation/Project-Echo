//
// Audio data example
// Minosha Gamage
//

interface Audio {ID:number; species:string; recording:number; timestamp:number; latitude:number; longitude:number};

let data:[Audio] = <[Audio]>[];	// start with empty array
let position:number = -1;       // current position 


function doAdd():void {
	document.getElementById("message").innerHTML = "";//clear message
	let idString: string = (<HTMLInputElement>document.getElementById("AudioID")).value;
	let id:number = parseInt(idString);
	if (isNaN(id)) {
		document.getElementById("message").innerHTML =
        		"Audio ID must be a number!"; // error message
		return; // do nothing
	}
	let species: string = (<HTMLInputElement>document.getElementById("Species")).value;
	if (species == "") {
		document.getElementById("message").innerHTML =
        		"Species must be entered"; 
		return; // do nothing
	}
	// we can add an element to end of data
	
	position = data.length;  // one past end of array
	data[position] = { ID: id, Species: species, Recording: recording, Timestamp: timestamp, Latitude: latitude, Longitude: longitude};
	
	updateDisplay();
}

function doNext() {
	if (position >= data.length-1) {
		document.getElementById("message").innerHTML =
        		"Already at end of list"; // error message
		return; // do nothing
	}
	
	position++;  // back up position
	updateDisplay();
}

function doPrev() {
	if (position==0) {
		document.getElementById("message").innerHTML =
        		"Already at start of list"; // error message
		return; // do nothing
	}
	
	position--;  // back up position
	updateDisplay();
}

function doDelete() {
	if (position < 0) {  // check for empty list
		document.getElementById("message").innerHTML =
        		"List is empty"; // error message
		return; // do nothing
	}
	// delete array element at current position
	data.splice(position, 1);  // tricky javascript code
	if (data.length == position) position--; // deleted last element in list
	updateDisplay();
}

function updateDisplay():void {
	let posString:string = "";
	if (position >= 0) {	// there is some data
		let current:Audio  = data[position];
		(<HTMLInputElement>document.getElementById("AudioID")).value = ""+current.ID;
		(<HTMLInputElement>document.getElementById("Species")).value = current.Species;
		(<HTMLInputElement>document.getElementById("Recording")).value = ""+current.Recording;
		(<HTMLInputElement>document.getElementById("Timestamp")).value = current.Timestamp;
		(<HTMLInputElement>document.getElementById("Latitude")).value = ""+current.Latitude;
		(<HTMLInputElement>document.getElementById("Longitude")).value = current.Longitude;
		posString = "record " + (position+1) + " of " + data.length;
	}  else {  // list is empty
		(<HTMLInputElement>document.getElementById("AudioID")).value = "";
		(<HTMLInputElement>document.getElementById("Species")).value = "";
		(<HTMLInputElement>document.getElementById("Recording")).value = "";
		(<HTMLInputElement>document.getElementById("Timestamp")).value = "";
		(<HTMLInputElement>document.getElementById("Latitude")).value = "";
		(<HTMLInputElement>document.getElementById("Longitude")).value = "";
		posString = "Audio list is empty";
	}
	document.getElementById("currentRecord").innerHTML = posString;
}
