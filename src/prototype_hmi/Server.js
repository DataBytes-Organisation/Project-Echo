//
// Simple web server for testing stuff
// Minosha Gamage

var port=8080;		// http port for listening
console.log("Listening on port " + port + ".");

var http = require('http');
var url = require('url');
var fs = require('fs');
var querystring = require('querystring');

var requests = 0;		// keep count of requests

var server = http.createServer(function (req,res) {
	requests++;
	console.log(">>> "+requests+": "+req.url);	// log the request URL
	
	var pathname = url.parse(req.url).pathname;  // extract the path name from the URL
	pathname = "." + pathname;                   // public data is in same folder as server
	
	if (req.method == 'POST') {  				// also log HTTP POST data on console
		var data = '';
		req.on('data',function (d) {			// save all data events
			data += d;
		});
		req.on('end', function () {				// when end of data write to console
			console.log(data);
		});
	}
	
	// now form the HTTP response to send to the client
	fs.stat(pathname, function (err,stat) {
		if (err) {
			console.log(err.code+": "+err.path);  				//       log all of the error object
			res.writeHead(404, {'Content-type' : 'text/html'});
			res.end();
			return;
		} else if (stat.isDirectory()) {
			pathname+="/index.html";
		}
		fs.readFile(pathname, function(err,data) {		// does file exist?
			if (err) { 									// NO->  bad path name
				console.log(err.code+": "+err.path);  						//       log all of the error object
				res.writeHead(404, {'Content-type' : 'text/html'});
			} else {   									// YES-> file exists so send it back to client
				if (pathname.substr(-4) == ".css" ) {
					res.writeHead(200, {'Content-type' : 'text/css'});
				} else {
					res.writeHead(200, {'Content-type' : 'text/html'});
				}
				res.write(data);
			}
		res.end();
		});
	});
}).listen(port);	// listen on configured port
//
// print some interface information
//
var ipList = require('os').networkInterfaces();		// list of interfaces
for (var device in ipList) { 						// for each device in the list of interfaces 
   var ipData=ipList[device];
   console.log("Interface "+device);				// header for each device	
   for (var j=0 ; j<ipData.length; j++) {			// for each interface of device print infor
	   console.log("    "+(ipData[j].internal ? "INTERNAL " : "EXTERNAL ")+
	      ipData[j].family + ": " +
		  ipData[j].address);
   }
}