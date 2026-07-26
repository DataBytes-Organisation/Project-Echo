//API to handle audio recording 
export function getAudioRecorder(){

var audioRecorder = {

    audioBlobs: [],
    mediaRecorder: null, 
    streamBeingCaptured: null, 

    start: function () {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate:48000});

        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
            return Promise.reject(new Error('mediaDevices API or getUserMedia method is not supported in this browser.'));
        }

        else {
            /*return navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
              // Create an audio source from the microphone stream
              const audioSource = audioContext.createMediaStreamSource(stream);

              // Create an AudioWorkletNode for resampling
              audioContext.audioWorklet.addModule('resampler-processor.js').then(() => {
                const resamplerNode = new AudioWorkletNode(audioContext, 'resampler-processor');

                // Set the desired sample rate (e.g., 16000 Hz)
                resamplerNode.port.postMessage({ targetSampleRate: 16000 });

                // Connect the audio source to the resampler
                audioSource.connect(resamplerNode);

                // Connect the resampler to the MediaRecorder
                const mediaRecorder = new MediaRecorder(resamplerNode.stream);

                console.log(mediaRecorder.audioBitsPerSecond);

                mediaRecorder.addEventListener("dataavailable", event => {
                  audioRecorder.audioBlobs.push(event.data);
                });

                mediaRecorder.start();
              });

            })
            .catch(err => console.error("Error accessing microphone: ", err));*/
            
            return navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    audioRecorder.streamBeingCaptured = stream;
                    audioRecorder.mediaRecorder = new MediaRecorder(stream);

                    console.log(audioRecorder.mediaRecorder.audioBitsPerSecond)

                    audioRecorder.audioBlobs = [];

                    audioRecorder.mediaRecorder.addEventListener("dataavailable", event => {
                        audioRecorder.audioBlobs.push(event.data);
                    });
                    
                    audioRecorder.mediaRecorder.addEventListener("stop", () => {
      
                        const audioBlob = new Blob(audioRecorder.audioBlobs);
                        
                        var buf = audioRecorder.resample(audioBlob);
                        
                        const audioUrl = URL.createObjectURL(audioBlob);
                  
                        const audio = new Audio(audioUrl);
                        audio.play();
                    });

                    audioRecorder.mediaRecorder.start();
                })
                .catch(err => console.error("Error accessing microphone: ", err));
        }
    },

    resample: function (audioChunks) {
        var audioCtx = new (AudioContext || webkitAudioContext)();
        
        
        var reader1 = new FileReader();
        reader1.onload = function(ev) {
            
            // Decode audio
            audioCtx.decodeAudioData(ev.target.result).then(function(buffer) {
      
              // Process Audio
              var offlineAudioCtx = new OfflineAudioContext({
                numberOfChannels: 2,
                length: 44100 * buffer.duration,
                sampleRate: 44100,
              });
      
              // Audio Buffer Source
              var soundSource = offlineAudioCtx.createBufferSource();
              soundSource.buffer = buffer;
      
              // // Create Compressor Node
              var compressor = offlineAudioCtx.createDynamicsCompressor();
      
              compressor.threshold.setValueAtTime(-20, offlineAudioCtx.currentTime);
              compressor.knee.setValueAtTime(30, offlineAudioCtx.currentTime);
              compressor.ratio.setValueAtTime(5, offlineAudioCtx.currentTime);
              compressor.attack.setValueAtTime(.05, offlineAudioCtx.currentTime);
              compressor.release.setValueAtTime(.25, offlineAudioCtx.currentTime);
      
      
      
              // Gain Node
              var gainNode = offlineAudioCtx.createGain();
              gainNode.gain.setValueAtTime(1, offlineAudioCtx.currentTime);
              
              // Connect nodes to destination
              soundSource.connect(compressor);
              compressor.connect(gainNode);
              gainNode.connect(offlineAudioCtx.destination);
      
                var reader2 = new FileReader();
      
      
               console.log("Created Reader2");
      
               reader2.onload = function(ev) {
      
                  console.log("Reading audio data to buffer...");
      
      
      
      
                  offlineAudioCtx.startRendering().then(function(renderedBuffer) {
                    // console.log('Rendering completed successfully.');
                        
                    var song = offlineAudioCtx.createBufferSource();
      
                    console.log('Rendered buffer:')
                    console.log(renderedBuffer);
      
                    console.log('OfflineAudioContext: ');
                    console.log(offlineAudioCtx);
                    
                    audioRecorder.make_download(renderedBuffer, offlineAudioCtx.length);
      
      
                  }).catch(function(err) {
                    console.log('Rendering failed: ' + err);
                  });
      
                  soundSource.loop = false;
              };
              
              reader2.readAsArrayBuffer(audioChunks);
                soundSource.start(0);
              
            });
          };
          reader1.readAsArrayBuffer(audioChunks); 
      },

      make_download: function (abuffer, total_samples) {

        // set sample length and rate
        var duration = abuffer.duration,
          rate = abuffer.sampleRate,
          offset = 0;
      
        console.log('abuffer');
        console.log(abuffer);
        var blob = audioRecorder.bufferToWave(abuffer, total_samples);
      
      
        console.log('blob');
        console.log(blob);
        // Generate audio file and assign URL
        var new_file = URL.createObjectURL(blob);

        const filename = prompt("Enter a filename for the JSON file:", "data.json");

        if (filename) {
      
          const downloadLink = document.createElement('a');
          downloadLink.href = new_file;
          downloadLink.download = filename;
          downloadLink.textContent = 'Download JSON';
      
          document.getElementById("downloadLink").innerHTML = "";
          document.getElementById("downloadLink").appendChild(downloadLink);
          downloadLink.click();
        }
      },
      
      
      // Convert AudioBuffer to a Blob using WAVE representation
      bufferToWave: function (abuffer, len) {
        var numOfChan = abuffer.numberOfChannels,
        length = len * numOfChan * 2 + 44,
        buffer = new ArrayBuffer(length),
        view = new DataView(buffer),
        channels = [], i, sample,
        offset = 0,
        pos = 0;
      
        // write WAVE header
        setUint32(0x46464952);                         // "RIFF"
        setUint32(length - 8);                         // file length - 8
        setUint32(0x45564157);                         // "WAVE"
      
        setUint32(0x20746d66);                         // "fmt " chunk
        setUint32(16);                                 // length = 16
        setUint16(1);                                  // PCM (uncompressed)
        setUint16(numOfChan);
        setUint32(abuffer.sampleRate);
        setUint32(abuffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
        setUint16(numOfChan * 2);                      // block-align
        setUint16(16);                                 // 16-bit (hardcoded in this demo)
      
        setUint32(0x61746164);                         // "data" - chunk
        setUint32(length - pos - 4);                   // chunk length
      
        // write interleaved data
        for(i = 0; i < abuffer.numberOfChannels; i++)
          channels.push(abuffer.getChannelData(i));
      
        while(pos < length) {
          for(i = 0; i < numOfChan; i++) {             // interleave channels
            sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
            sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767)|0; // scale to 16-bit signed int
            view.setInt16(pos, sample, true);          // write 16-bit sample
            pos += 2;
          }
          offset++                                     // next source sample
        }
      
        // create Blob
        return new Blob([buffer], {type: "audio/wav"});
      
        function setUint16(data) {
          view.setUint16(pos, data, true);
          pos += 2;
        }
      
        function setUint32(data) {
          view.setUint32(pos, data, true);
          pos += 4;
        }
      },

    stop: function () {
        return new Promise(resolve => {
            let mimeType = 'audio/wav; codecs=MS_PCM'
            //let mimeType = audioRecorder.mediaRecorder.mimeType;

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