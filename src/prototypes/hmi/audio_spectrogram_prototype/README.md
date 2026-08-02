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
