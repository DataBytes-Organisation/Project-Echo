# Audio spectrogram prototype

This folder contains the Project Echo FR-B1 HMI prototype. It renders decoded
audio as a canvas spectrogram in animal and microphone panel layouts.

## Requirements

- Node.js 22 (the prototype was verified with Node.js 22.18.0)
- A browser with Canvas, Web Audio, and ResizeObserver support

The prototype has no external npm dependencies, so `npm install` is not
required.

## Run the automated checks

Open PowerShell in this folder, or change to it from the repository root:

```powershell
Set-Location 'src\prototypes\hmi\audio_spectrogram_prototype'

npm test
npm run check
```

`npm test` runs the Node test suite for audio decoding boundaries, FFT output,
canvas drawing, panel resizing, visible states, load races, and cleanup.
`npm run check` validates the JavaScript syntax.

To run only the tests that import the production HMI modules:

```powershell
node --test tests/production-spectrogram.test.js
```

These tests stay in this prototype folder but load source from
`src/production/hmi/ui/public/js/`. They cover production DSP, canvas drawing,
decode boundaries, panel resize behavior, states, source races, and cleanup.

## Run the browser prototype

```powershell
$env:PORT = '4173'
npm start
```

Open <http://127.0.0.1:4173/>. Stop the server with `Ctrl+C`.

If port 4173 is already in use, select another local port:

```powershell
$env:PORT = '4181'
npm start
```

## Manual checks

1. Confirm both panels show an empty state on first load.
2. Select `Generated clip` and confirm both panels render nonblank intensity
   cells with seconds, frequency, and decibel labels.
3. Select `Loading` and confirm both panels show the decoding state.
4. Select `Decode error` and confirm both panels show the public decode message.
5. Select `Local audio` and choose a browser-supported WAV, MP3, or similar
   audio file.
6. Resize the browser across desktop and tablet widths. The canvas should stay
   inside each panel without clipping or overlap.

The demo does not request a microphone and does not call production APIs, MQTT,
or other live services.

## Run the production UI locally

From the repository root, serve the production HMI public folder with Python:

```powershell
Set-Location 'src\production\hmi\ui\public'
python -m http.server 4182 --bind 127.0.0.1
```

Open <http://127.0.0.1:4182/index.html>. Stop the server with `Ctrl+C`.

Without the Project Echo backend, the page can still verify the microphone
panel's empty, loading, success, decode-error, and responsive states:

1. Open the microphone panel and confirm its initial empty state.
2. Choose a browser-supported WAV, MP3, OGG, or WebM audio file and confirm a
   nonblank spectrogram appears with time, frequency, and decibel context.
3. Choose a text or corrupt file and confirm the loading state is replaced by
   a short decode error with no technical response details.
4. Resize the browser to desktop and tablet widths and confirm the canvas stays
   inside the panel and remains sharp.
5. Grant microphone permission, record a short clip, stop recording, and
   confirm the recorded clip produces a spectrogram and remains playable.

Animal audio requires the production HMI's `/audio/:id` backend route. When
that backend is unavailable, selecting an animal should show the public
unavailable message instead of a blank canvas or internal error response.
