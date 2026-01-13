[<- Back to Main Guide](model_integration_guide.md)

---

## What is Preprocessing and Why is it Necessary?

The classifier, a Convolutional Neural Network (CNN), is designed to process 2D data structures, similar to images. Raw audio, being a 1D waveform, requires transformation. The goal of preprocessing is to convert the 1D sound wave into a 2D, image-like format that a CNN can effectively analyse. This format is the Mel Spectrogram.

The creation process is as follows:

1.  **Create a Spectrogram (The "Audio Image")**:
    *   A spectrogram decomposes the sound into its constituent frequencies over time. This is analogous to creating an audio image from an audio file, where the image shows which notes (frequencies) are played at what times and with what intensity (loudness). The result is a chart with time on one axis, frequency on the other, and the colour/brightness of each point showing the loudness of that frequency at that moment.

2.  **Apply the "Mel" Scale**:
    *   Humans perceive sound frequencies non-linearly; we are more sensitive to changes in low frequencies than high ones.
    *   The **Mel scale** reorganises the spectrogram's frequency axis to mimic this perception. It allocates more resolution to lower frequencies and compresses higher frequencies. This helps the model focus on the frequency ranges most critical for distinguishing real-world sounds.

3.  **Use Decibels & Normalisation (Standardise the "Image")**:
    *   Loudness values are converted to a logarithmic scale (decibels), which aligns better with human hearing.
    *   Finally, all values are normalised to a standard range (e.g., 0 to 1). This is like adjusting the brightness and contrast of a photo, ensuring the final "image" is clear and consistent for the model, regardless of the original recording's volume.

In essence, the preprocessor transforms a 1D sound wave into a standardised 2D "image" (the Mel Spectrogram) optimised for both human auditory perception and the pattern-recognition capabilities of a CNN.

---
[<- Back to Main Guide](model_integration_guide.md)
