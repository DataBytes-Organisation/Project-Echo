const DEFAULT_AUDIO_PATH = './audio/default.mp3';

function getColor(v) {
    v = Math.pow(Math.max(0, Math.min(1, v)), 0.5);
    let r, g, b;

    if (v < 0.25) {
        r = 0; g = 0; b = 255 * (v / 0.25);
    } else if (v < 0.5) {
        r = 255 * ((v - 0.25) / 0.25); g = 0; b = 255;
    } else if (v < 0.75) {
        r = 255; g = 255 * ((v - 0.5) / 0.25); b = 0;
    } else {
        r = 255; g = 255; b = 255 * ((v - 0.75) / 0.25);
    }

    return `rgb(${r|0},${g|0},${b|0})`;
}

function fft(re, im) {
    const N = re.length;
    if (N <= 1) return;

    const half = N / 2;
    const evenRe = new Float32Array(half);
    const evenIm = new Float32Array(half);
    const oddRe = new Float32Array(half);
    const oddIm = new Float32Array(half);

    for (let i = 0; i < half; i++) {
        evenRe[i] = re[i*2];
        evenIm[i] = im[i*2];
        oddRe[i] = re[i*2+1];
        oddIm[i] = im[i*2+1];
    }

    fft(evenRe, evenIm);
    fft(oddRe, oddIm);

    for (let k = 0; k < half; k++) {
        const angle = (-2 * Math.PI * k) / N;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        const tre = cos * oddRe[k] - sin * oddIm[k];
        const tim = sin * oddRe[k] + cos * oddIm[k];

        re[k] = evenRe[k] + tre;
        im[k] = evenIm[k] + tim;
        re[k + half] = evenRe[k] - tre;
        im[k + half] = evenIm[k] - tim;
    }
}

async function generateSpectrogram(src, canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const response = await fetch(src);
    const arrayBuffer = await response.arrayBuffer();

    const audioCtx = new AudioContext();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const data = audioBuffer.getChannelData(0);

    const fftSize = 2048;
    const hopSize = 512;
    const bins = fftSize / 2;
    const totalFrames = Math.floor((data.length - fftSize) / hopSize);
    const sliceWidth = canvas.width / totalFrames;

    const hannWindow = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++) {
        hannWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (fftSize - 1)));
    }

    const real = new Float32Array(fftSize);
    const imag = new Float32Array(fftSize);

    for (let frame = 0; frame < totalFrames; frame++) {
        const offset = frame * hopSize;

        for (let i = 0; i < fftSize; i++) {
            real[i] = (data[offset + i] || 0) * hannWindow[i];
            imag[i] = 0;
        }

        fft(real, imag);

        for (let i = 0; i < bins; i++) {
            const mag = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
            const intensity = Math.log10(1 + mag) / Math.log10(1000);
            const logIndex = Math.log10(1 + 9 * (i / bins));
            const y = canvas.height - logIndex * canvas.height;

            ctx.fillStyle = getColor(intensity);
            ctx.fillRect(frame * sliceWidth, y, sliceWidth, canvas.height / bins);
        }
    }
}

export async function initSpectrogram(rawPath) {
    const src = rawPath ?? DEFAULT_AUDIO_PATH;

    const audio = document.getElementById("spectrogram-audio");
    const canvas = document.getElementById("spectrogram-canvas");
    const playhead = document.getElementById("spectrogram-playhead");
    const playButton = document.getElementById("spectrogram-play-btn");
    const container = document.getElementById("spectrogram-container");

    // Reset
    audio.pause();
    audio.src = src;
    audio.load();
    playhead.style.left = "0px";

    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;

    audio.addEventListener("loadeddata", async () => {
        await generateSpectrogram(src, canvas);
    }, { once: true });

    playButton.addEventListener("click", (e) => {
        e.stopPropagation();
        if (audio.paused) {
            audio.play();
        } else {
            audio.pause();
        }
    }, { once: false });

    container.addEventListener("click", (e) => {
        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percent = x / rect.width;
        audio.currentTime = percent * audio.duration;
    });

    audio.addEventListener("play", () => {
        function frame() {
            if (!audio.duration) return;
            const percent = audio.currentTime / audio.duration;
            playhead.style.left = (percent * canvas.width) + "px";
            if (!audio.paused) requestAnimationFrame(frame);
        }
        frame();
    });
}