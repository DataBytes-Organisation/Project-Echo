//API to handle audio recording 
export function getAudioRecorder(){

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
                    audioRecorder.streamBeingCaptured = stream;
                    audioRecorder.mediaRecorder = new MediaRecorder(stream);
                    audioRecorder.audioBlobs = [];

                    audioRecorder.mediaRecorder.addEventListener("dataavailable", event => {
                        audioRecorder.audioBlobs.push(event.data);
                    });

                    audioRecorder.mediaRecorder.start();
                })
                .catch(err => console.error("Error accessing microphone: ", err));
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

    stopStream: function () {
        audioRecorder.streamBeingCaptured.getTracks() 
            .forEach(track => track.stop()); 
    },

    resetRecordingProperties: function () {
        audioRecorder.mediaRecorder = null;
        audioRecorder.streamBeingCaptured = null;

    }
}

return audioRecorder;

}