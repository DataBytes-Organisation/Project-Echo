import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile } from "@ffmpeg/util";
import FFT from "fft.js";

const ffmpeg = new FFmpeg();

const convertToWav = async (file, onProgress) => {
  if (!ffmpeg.loaded) {
    await ffmpeg.load();
  }

  const fileType = file.name.split(".").pop().toLowerCase();
  const inputFileName = `input.${fileType}`;
  const outputFileName = "output.wav";

  try {
    // Write the file to FFmpeg's file system
    await ffmpeg.writeFile(inputFileName, await fetchFile(file));

    // Run FFmpeg command
    await ffmpeg.exec(["-i", inputFileName, outputFileName]);

    // Read the output file
    const data = await ffmpeg.readFile(outputFileName);
    const blob = new Blob([data.buffer], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);

    if (onProgress) {
      onProgress(100);
    }

    return url;
  } catch (error) {
    console.error("Error in convertToWav:", error);
    throw error;
  }
};

const convertToWavChunk = async (file, start, end, onProgress) => {
  if (!ffmpeg.loaded) {
    await ffmpeg.load();
  }

  const fileType = file.name.split(".").pop().toLowerCase();
  const inputFileName = `input.${fileType}`;
  const outputFileName = "output_chunk.wav";

  try {
    await ffmpeg.writeFile(inputFileName, await fetchFile(file));
    await ffmpeg.exec([
      "-i",
      inputFileName,
      "-ss",
      `${start}`,
      "-to",
      `${end}`,
      "-ar",
      "44100",
      "-ac",
      "2",
      "-c:a",
      "pcm_s16le",
      outputFileName,
    ]);

    const data = await ffmpeg.readFile(outputFileName);
    const blob = new Blob([data.buffer], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);

    if (onProgress) {
      onProgress(100);
    }

    console.log("Generated WAV URL:", url); // Debug: Log the generated URL

    return url;
  } catch (error) {
    console.error("Error in convertToWavChunk:", error);
    throw error;
  }
};

const generateSpectrogram = (audioBuffer) => {
  const fft = new FFT(1024);
  const out = fft.createComplexArray();
  const input = new Float32Array(audioBuffer);

  fft.realTransform(out, input);
  fft.completeSpectrum(out);

  const spectrogramData = [];
  for (let i = 0; i < out.length; i += 2) {
    const magnitude = Math.sqrt(out[i] ** 2 + out[i + 1] ** 2);
    spectrogramData.push(magnitude);
  }

  return spectrogramData;
};

const convertToWavWithSpectrogram = async (file, start, end, onProgress) => {
  const chunkUrl = await convertToWavChunk(file, start, end, onProgress);

  // Load the WAV file using fetch and decode audio data
  const response = await fetch(chunkUrl);
  const arrayBuffer = await response.arrayBuffer();

  // Create an AudioContext to decode the audio data
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  // Convert audio data to Float32Array for FFT
  const channelData = audioBuffer.getChannelData(0);
  const spectrogramData = generateSpectrogram(channelData.buffer);

  return { chunkUrl, spectrogramData };
};

export { convertToWav, convertToWavChunk, convertToWavWithSpectrogram };
