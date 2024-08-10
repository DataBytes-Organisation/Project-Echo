"use client";


import * as tf from "@tensorflow/tfjs";

class SimpleNAS {
  constructor(dataFile = null, modelFile = null) {
    this.dataFile = dataFile;
    this.modelFile = modelFile;
    this.spectrogram = null;
    this.labels = null;
    this.spectrogramTensor = null;
    this.labelsTensor = null;
    this.model = null;
  }

  async loadData() {
    try {
      const response = await fetch(this.dataFile);
      const jsonData = await response.json();
      this.spectrogram = jsonData.spectrogram;
      this.labels = jsonData.labels.map((label) =>
        label === "" ? 0 : parseInt(label)
      );
    } catch (error) {
      console.error("Error loading data:", error);
    }
  }

  async importModel() {
    if (this.modelFile) {
      try {
        this.model = await tf.loadLayersModel(this.modelFile);
        console.log("Model loaded successfully.");
      } catch (error) {
        console.error("Error loading model:", error);
      }
    } else {
      console.log("No model file provided. Building a new model.");
      this.buildModel();
    }
  }

  prepareTensors() {
    this.spectrogramTensor = tf
      .tensor2d(this.spectrogram, [1, this.spectrogram.length])
      .expandDims(2);

    this.labelsTensor = tf.tensor1d([this.labels[0]], "float32");
  }

  buildModel() {
    this.model = tf.sequential();

    this.model.add(
      tf.layers.conv1d({
        inputShape: [1024, 1],
        filters: 8,
        kernelSize: 3,
        activation: "relu",
      })
    );

    this.model.add(tf.layers.maxPooling1d({ poolSize: 2 }));
    this.model.add(
      tf.layers.conv1d({ filters: 16, kernelSize: 3, activation: "relu" })
    );
    this.model.add(tf.layers.maxPooling1d({ poolSize: 2 }));

    this.model.add(tf.layers.flatten());
    this.model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

    this.model.compile({
      optimizer: "adam",
      loss: "sparseCategoricalCrossentropy",
      metrics: ["accuracy"],
    });
  }

  async trainModel() {
    if (!this.model) {
      this.buildModel();
    }

    await this.model.fit(this.spectrogramTensor, this.labelsTensor, {
      epochs: 10,
    });
    console.log("Model training complete.");
  }

  async saveModel() {
    await this.model.save("downloads://cnn-model");
    console.log("Model saved.");
  }

  getModel() {
    return this.model.toJSON();
  }

  setModel(model) {
    this.model = tf.models.modelFromJSON(model);
  }

  padInput(inputArray, targetLength) {
    if (inputArray.length < targetLength) {
      // Pad with zeros
      return [
        ...inputArray,
        ...new Array(targetLength - inputArray.length).fill(0),
      ];
    } else if (inputArray.length > targetLength) {
      // Truncate if it's too long
      return inputArray.slice(0, targetLength);
    }
    return inputArray;
  }

  async predict(data) {
    const paddedData = this.padInput(data.spectrogram, 1024);
    const tensor = tf.tensor2d(paddedData, [1, 1024]).expandDims(2);
    const prediction = this.model.predict(tensor);
    return prediction.arraySync();
  }
}

export default SimpleNAS;
