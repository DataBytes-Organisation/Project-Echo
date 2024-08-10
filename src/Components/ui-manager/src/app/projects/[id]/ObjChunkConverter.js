"use client";


import React, { Component } from "react";
import {
  Box,
  VStack,
  Text,
  Flex,
  Spacer,
  Button,
  Checkbox,
  Progress,
  Wrap,
  WrapItem,
  Stack,
  Card,
  CardHeader,
  Heading,
  StackDivider,
  CardBody,
} from "@chakra-ui/react";

import md5 from "md5";

import { convertToWavWithSpectrogram } from "./audioUtils";

class ObjChunkConverter extends Component {
  constructor(props) {
    super(props);
    this.state = {
      chunkUrl: null,
      chunkDuration: 0,
      isPlaying: false,
      labeledData: new Map(),
      generatingAll: false,
      progress: 0,
    };
  }

  handleGenerateAllSpectrograms = async () => {
    const { mapChunkWithLabel } = this.props;
    const chunkEntries = Array.from(mapChunkWithLabel.entries());
    this.setState({ generatingAll: true, progress: 0 });

    for (let i = 0; i < chunkEntries.length; i++) {
      const [chunkId, chunkData] = chunkEntries[i];
      const key = `${chunkData.uniqPassFile.name}_${chunkId}`;
      if (!this.state.labeledData.has(key)) {
        console.log(`Generating spectrogram for: ${key}`);
        await this.handleGenerateSpectrogram(chunkData, true);
        this.setState({ progress: ((i + 1) / chunkEntries.length) * 100 });
      }
    }

    this.setState({ generatingAll: false });
  };

  handleGenerateSpectrogram = async (chunkData, isBatch = false) => {
    const { uniqPassFile, uniqPassIn, lstCats } = chunkData;
    const start = uniqPassIn.start;
    const end = uniqPassIn.end;

    try {
      const { chunkUrl, spectrogramData } = await convertToWavWithSpectrogram(
        uniqPassFile,
        start,
        end,
        null
      );
      console.log(`Converted to WAV and generated spectrogram: ${chunkUrl}`);

      const spectrogramMD5 = this.generateMD5(spectrogramData);
      console.log(
        `MD5 Hash for ${uniqPassFile.name}_${uniqPassIn.id}: ${spectrogramMD5}`
      );

      const key = `${uniqPassFile.name}_${uniqPassIn.id}`;
      this.setState((prevState) => {
        const updatedMap = new Map(prevState.labeledData);
        console.log({
          spectrogram: spectrogramData,
          labels: lstCats,
          md5: spectrogramMD5,
          uploaded: false,
        });
        updatedMap.set(key, {
          spectrogram: spectrogramData,
          labels: lstCats,
          md5: spectrogramMD5,
          uploaded: false,
        });
        return { labeledData: updatedMap };
      });

      console.log(`Spectrogram data extracted for: ${key}`);
    } catch (error) {
      console.error("Error generating spectrogram:", error);
    }
  };

  handleUpload = async (key) => {
    const { projectId } = this.props;
    const chunkData = this.state.labeledData.get(key);

    console.log(`Project ID: ${projectId}`);
    console.log(`Chunk Data:`, chunkData);

    if (!chunkData || !chunkData.labels || !chunkData.spectrogram) {
      console.error("Invalid chunk data:", chunkData);
      return;
    }

    const formData = new FormData();
    const labelsJson = JSON.stringify(chunkData.labels);
    const spectrogramJson = JSON.stringify(chunkData.spectrogram);

    console.log(`Uploading with labels: ${labelsJson}`);
    console.log(`Uploading with spectrogram: ${spectrogramJson}`);

    formData.append("project_id", projectId);
    formData.append("labels", labelsJson);
    formData.append("spectrogram", spectrogramJson);

    try {
      const response = await fetch("/pback/upload/spectrogram", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorStatus = response.status;
        const errorText = await response.text(); // Extract error message
        throw new Error(
          `Error uploading spectrogram: ${errorStatus} - ${errorText}`
        );
      }

      const result = await response.json();
      this.setState((prevState) => {
        const updatedMap = new Map(prevState.labeledData);
        const updatedChunkData = { ...updatedMap.get(key), uploaded: true };
        updatedMap.set(key, updatedChunkData);
        return { labeledData: updatedMap };
      });
      console.log(`Uploaded successfully: ${key}`);
    } catch (error) {
      console.log("Error uploading spectrogram:", error);
      // Display a user-friendly error message using state management or error toasts
    }
  };

  generateMD5 = (data) => {
    return md5(JSON.stringify(data));
  };

  drawSpectrogram = (canvas, spectrogramData) => {
    const context = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const imageData = context.createImageData(width, height);
    const data = imageData.data;

    // Find max and min values in the spectrogramData for normalization
    const maxVal = Math.max(...spectrogramData);
    const minVal = Math.min(...spectrogramData);

    // Adjust the contrast and color mapping
    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        const index = (x + y * width) * 4;
        const value = spectrogramData[x * height + y];
        const normalizedValue = ((value - minVal) / (maxVal - minVal)) * 255;
        data[index] = normalizedValue;
        data[index + 1] = normalizedValue;
        data[index + 2] = normalizedValue;
        data[index + 3] = 255; // Alpha channel
      }
    }

    context.putImageData(imageData, 0, 0);
  };

  render() {
    const { mapChunkWithLabel } = this.props;
    const { generatingAll, progress, labeledData } = this.state;

    return (
      <Box>
        <Button
          onClick={this.handleGenerateAllSpectrograms}
          disabled={generatingAll}
        >
          {generatingAll ? "Generating..." : "Generate All Spectrograms"}
        </Button>
        {generatingAll && <Progress value={progress} size="sm" mt={2} />}
        <Wrap spacing={4} align="stretch" mt={4}>
          {Array.from(mapChunkWithLabel.entries()).map(
            ([chunkId, chunkData]) => {
              const key = `${chunkData.uniqPassFile.name}_${chunkId}`;
              const isChecked = labeledData.has(key);
              const isUploaded = labeledData.get(key)?.uploaded;
              const md5Hash = labeledData.get(key)?.md5;
              const spectrogramData = labeledData.get(key)?.spectrogram;
              return (
                <WrapItem
                  key={chunkId}
                  p={4}
                  borderWidth="1px"
                  borderRadius="lg"
                  bg="gray.700"
                >
                  <Card bg="transparent" color={"white"}>
                    <CardBody>
                      <Stack divider={<StackDivider />} spacing="4">
                        <Box>
                          <Heading size="xs" textTransform="uppercase">
                            {chunkId}: {chunkData.uniqPassIn.start}s -{" "}
                            {chunkData.uniqPassIn.end}
                          </Heading>
                          <Text pt="2" fontSize="sm">
                            {md5Hash}
                          </Text>
                        </Box>
                        <Box>
                          <Heading size="xs" textTransform="uppercase">
                            File: {chunkData.uniqPassFile.name}
                          </Heading>
                          <Stack>
                            <Text>Category: {chunkData.lstCats[0]}</Text>
                            <Text>Subcategory: {chunkData.lstCats[1]}</Text>
                            <Text>Sub-subcategory: {chunkData.lstCats[2]}</Text>
                          </Stack>
                        </Box>
                        <Box>
                          <Heading size="xs" textTransform="uppercase">
                            <Checkbox isChecked={isChecked} isDisabled>
                              Processed
                            </Checkbox>
                          </Heading>

                          <Button
                            onClick={() =>
                              this.handleGenerateSpectrogram(chunkData)
                            }
                            disabled={isChecked}
                          >
                            {isChecked
                              ? "Already Generated"
                              : "Generate Spectrogram"}
                          </Button>
                          <Button
                            onClick={() => this.handleUpload(key)}
                            disabled={!isChecked || isUploaded}
                            colorScheme={isUploaded ? "green" : "red"}
                          >
                            {isUploaded ? "Already Uploaded" : "Upload"}
                          </Button>
                        </Box>
                        {spectrogramData && (
                          <Box>
                            <Heading size="xs" textTransform="uppercase">
                              Spectrogram
                            </Heading>
                            <canvas
                              ref={(canvas) =>
                                canvas &&
                                this.drawSpectrogram(canvas, spectrogramData)
                              }
                              width={256}
                              height={256}
                            />
                          </Box>
                        )}
                      </Stack>
                    </CardBody>
                  </Card>
                </WrapItem>
              );
            }
          )}
        </Wrap>
      </Box>
    );
  }
}

export default ObjChunkConverter;
