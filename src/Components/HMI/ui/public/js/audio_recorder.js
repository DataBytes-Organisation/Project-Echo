//API to handle audio recording 

var audioRecorder = {

    audioBlobs: [],
    mediaRecorder: null, 
    streamBeingCaptured: null, 

    start: function () {
        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
            return Promise.reject(new Error('mediaDevices API or getUserMedia method is not supported in this browser.'));
        }

        else {
            return navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {

                    //save the reference of the stream to be able to stop it when necessary
                    audioRecorder.streamBeingCaptured = stream;

                    //create a media recorder instance by passing that stream into the MediaRecorder constructor
                    audioRecorder.mediaRecorder = new MediaRecorder(stream); /*the MediaRecorder interface of the MediaStream Recording
                    API provides functionality to easily record media*/

                    //clear previously saved audio Blobs, if any
                    audioRecorder.audioBlobs = [];

                    //add a dataavailable event listener in order to store the audio data Blobs when recording
                    audioRecorder.mediaRecorder.addEventListener("dataavailable", event => {
                        //store audio Blob object
                        audioRecorder.audioBlobs.push(event.data);
                    });

                    //start the recording by calling the start method on the media recorder
                    audioRecorder.mediaRecorder.start();
                });
        }
    },

    stop: function () {
        return new Promise(resolve => {
            let mimeType = audioRecorder.mediaRecorder.mimeType;

            audioRecorder.mediaRecorder.addEventListener("stop", () => {
                let audioBlob = new Blob(audioRecorder.audioBlobs, { type: mimeType });
                resolve(audioBlob);
            });
            audioRecorder.cancel();
        });
    },

    cancel: function () {
        audioRecorder.mediaRecorder.stop();
        audioRecorder.stopStream();
        audioRecorder.resetRecordingProperties();
    },
    /** Stop all the tracks on the active stream in order to stop the stream and remove
     * the red flashing dot showing in the tab
     */
    stopStream: function () {
        audioRecorder.streamBeingCaptured.getTracks() 
            .forEach(track => track.stop()); 
    },

    resetRecordingProperties: function () {
        audioRecorder.mediaRecorder = null;
        audioRecorder.streamBeingCaptured = null;

    }
}