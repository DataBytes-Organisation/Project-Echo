"use client";

import React, { Component } from "react";

import { convertWholeWavToSpectrogram } from "./audioUtils"; // Make sure the path is correct

class ObjUpload extends Component {
  constructor(props) {
    super(props);
    this.state = {
      fileCount: 0,
      totalSizeMB: 0,
      files: [],
    };
  }

  // Upload function to post spectrogram data to the server
  uploadSpectrogram = async (file, spectrogramData) => {
    const formData = new FormData();

    const projectId = this.props.projectId; // Ensure this is coming from props
    const labels = this.props.labels || ["default_label"]; // Provide default labels if necessary

    if (!projectId) {
      console.error("Missing projectId in props.");
      return;
    }

    const spectrogramJson = JSON.stringify(spectrogramData);
    const labelsJson = JSON.stringify(labels);

    // Log to verify the values before sending
    console.log("Project ID:", projectId);
    console.log("Labels:", labelsJson);
    console.log("Spectrogram:", spectrogramJson);

    formData.append("project_id", projectId);
    formData.append("file_name", file.name);
    formData.append("spectrogram", spectrogramJson);
    formData.append("labels", labelsJson); // Ensure labels are sent

    try {
      const response = await fetch("/pback/upload/spectrogram", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorStatus = response.status;
        const errorText = await response.text();
        throw new Error(
          `Error uploading spectrogram: ${errorStatus} - ${errorText}`
        );
      }

      const result = await response.json();
      console.log(`Uploaded successfully: ${file.name}`, result);
    } catch (error) {
      console.error("Error uploading spectrogram:", error);
    }
  };

  handleFolderSelect = async (event) => {
    const files = event.target.files;
    let filesArray = [];
    let totalSize = 0;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const fileExtension = file.name.split(".").pop().toLowerCase(); // Extract the file extension

      if (fileExtension === "wav") {
        filesArray.push(file);
        totalSize += file.size; // Accumulate file size in bytes
        console.log("File:", file.name); // Log each file

        // Convert the .wav file and generate the spectrogram
        try {
          const { spectrogramData } = await convertWholeWavToSpectrogram(
            file,
            (progress) => {
              console.log(`Conversion progress for ${file.name}:`, progress);
            }
          );
          console.log(`Spectrogram data for ${file.name}:`, spectrogramData);

          // Upload the generated spectrogram
          await this.uploadSpectrogram(file, spectrogramData);
        } catch (error) {
          console.error(`Error converting ${file.name} to spectrogram:`, error);
        }
      } else {
        console.log("Not a .wav file:", file.name); // Log files that are not .wav
      }

      break; // If you want to handle only the first file
    }

    const totalFiles = filesArray.length;
    const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2); // Convert bytes to MB and round to 2 decimal places

    this.setState({
      fileCount: totalFiles,
      totalSizeMB: totalSizeMB,
      files: filesArray,
    });

    console.log("Total files to be uploaded:", totalFiles);
    console.log("Total size (MB):", totalSizeMB);
  };

  render() {
    return (
      <div>
        <input
          type="file"
          webkitdirectory="true"
          directory="true"
          onChange={this.handleFolderSelect}
          style={{ display: "block", marginBottom: "20px" }}
        />
        <div>Total Files to be uploaded: {this.state.fileCount}</div>
        <div>Total Size: {this.state.totalSizeMB} MB</div>
      </div>
    );
  }
}

export default ObjUpload;
