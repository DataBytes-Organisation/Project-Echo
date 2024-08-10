"use client";


import React from "react";
import {
  Box,
  Input,
  VStack,
  Text,
  Progress,
  Button,
  HStack,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Wrap,
  WrapItem,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Stack,
  StackDivider,
  Flex,
  Spacer,
} from "@chakra-ui/react";

import { toast } from "react-toastify";
import WaveSurfer from "wavesurfer.js";
import TimelinePlugin from "wavesurfer.js/dist/plugins/timeline";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions";
import { convertToWav } from "./audioUtils";
import withParams from "./withParams";

import ObjLabelManager from "../projects/[id]/ObjLabelManager";
import ObjSelectLabel from "./ObjSelectLabel";
import ObjChunkConverter from "../projects/[id]/ObjChunkConverter";
import ObjAiTrainer from "../projects/[id]/ObjAiTrainer";

class ObjProject extends React.Component {
  constructor(props) {
    super(props);
    const projectId = this.props.params.id;
    const savedProject = JSON.parse(
      localStorage.getItem(`project-${projectId}`)
    ) || {
      files: [],
      audioUrls: [],
      selectedFileIndex: null,
      chunks: [],
    };
    this.state = {
      ...savedProject,
      processing: false,
      progress: 0,
      error: null,
      isPlaying: false,
      volume: 0.5,
      zoom: 50,
      lstLabels: {},
      mapChunkWithLabel: new Map(),
      projectId: projectId,
    };
    this.waveSurferInstance = null;
    this.waveSurferRef = React.createRef();
    this.timelineRef = React.createRef();
    this.projectId = projectId;
  }

  componentDidUpdate() {
    localStorage.setItem(
      `project-${this.projectId}`,
      JSON.stringify(this.state)
    );
  }

  handleDataFromChildObjLabels = (data) => {
    this.setState({ lstLabels: data });
  };

  handleDataFromChildChunkWithLabel = (data) => {
    const { mapChunkWithLabel } = this.state;
    mapChunkWithLabel.set(data.uniqPassIn.id, data);
    this.setState({ mapChunkWithLabel: mapChunkWithLabel });
  };

  handleFileChange = async (e) => {
    const selectedFiles = Array.from(e.target.files);
    this.setState({
      files: selectedFiles,
      audioUrls: [],
      selectedFileIndex: null,
      processing: true,
      progress: 0,
      error: null,
    });

    const audioUrls = await this.processFiles(selectedFiles);
    this.setState({ audioUrls, processing: false, progress: 100 });
  };

  processFiles = async (files) => {
    const audioUrls = [];
    for (const file of files) {
      try {
        const audioUrl = await convertToWav(file, this.updateProgress);
        audioUrls.push(audioUrl);
      } catch (error) {
        console.error(`Error loading file ${file.name}:`, error);
        toast.error(`Error loading file ${file.name}: ${error.message}`, {
          position: "bottom-right",
          autoClose: 5000,
          hideProgressBar: false,
          closeOnClick: true,
          pauseOnHover: true,
          draggable: true,
          progress: undefined,
        });
      }
    }
    return audioUrls;
  };

  updateProgress = (progress) => {
    this.setState({ progress });
  };

  initWaveSurfer = (url) => {
    if (this.waveSurferInstance) {
      this.waveSurferInstance.destroy();
    }
    const regionsPlugin = RegionsPlugin.create();
    this.waveSurferInstance = WaveSurfer.create({
      container: this.waveSurferRef.current,
      waveColor: "violet",
      progressColor: "purple",
      responsive: true,
      height: 100,
      plugins: [
        TimelinePlugin.create({
          container: this.timelineRef.current,
        }),
        regionsPlugin,
      ],
    });
    this.waveSurferInstance.load(url);
    this.waveSurferInstance.on("play", () =>
      this.setState({ isPlaying: true })
    );
    this.waveSurferInstance.on("pause", () =>
      this.setState({ isPlaying: false })
    );
    this.waveSurferInstance.on("ready", () => {
      this.waveSurferInstance.setVolume(this.state.volume);
      this.waveSurferInstance.zoom(this.state.zoom);
      this.waveSurferInstance.plugins.regions = regionsPlugin;
    });
  };

  handlePlayPause = () => {
    if (this.waveSurferInstance) {
      this.waveSurferInstance.playPause();
    }
  };

  handleVolumeChange = (value) => {
    if (this.waveSurferInstance) {
      this.waveSurferInstance.setVolume(value);
      this.setState({ volume: value });
    }
  };

  handleZoomChange = (value) => {
    if (this.waveSurferInstance) {
      this.waveSurferInstance.zoom(value);
      this.setState({ zoom: value });
    }
  };

  addRegion = () => {
    if (this.waveSurferInstance && this.waveSurferInstance.plugins.regions) {
      this.waveSurferInstance.plugins.regions.addRegion({
        start: 1,
        end: 3,
        color: "rgba(0, 255, 0, 0.1)",
      });
    }
  };

  addMultipleRegions = () => {
    if (this.waveSurferInstance && this.waveSurferInstance.plugins.regions) {
      const regionsPlugin = this.waveSurferInstance.plugins.regions;
      const duration = this.waveSurferInstance.getDuration();
      const chunkSize = 1; // Fixed chunk size of 1 second

      for (let i = 0; i < duration; i += chunkSize) {
        const start = i;
        const end = i + chunkSize;
        regionsPlugin.addRegion({
          start,
          end,
          color: `rgba(0, 255, 0, 0.1)`,
        });
      }

      console.log(regionsPlugin);

      const regions = regionsPlugin.regions; // Updated to use regions array
      if (regions) {
        const chunks = regions.map((region, index) => ({
          id: region.id,
          start: region.start,
          end: region.end,
        }));
        console.log("Chunks:", chunks); // Log chunks to console
        this.setState({ chunks });
      }
    }
  };

  handleFileSelect = (index) => {
    this.setState({ selectedFileIndex: index }, () => {
      if (this.waveSurferRef.current && this.timelineRef.current) {
        try {
          this.initWaveSurfer(this.state.audioUrls[index]);
        } catch (error) {
          toast.error(`Error initializing WaveSurfer: ${error.message}`, {
            position: "bottom-right",
            autoClose: 5000,
            hideProgressBar: false,
            closeOnClick: true,
            pauseOnHover: true,
            draggable: true,
            progress: undefined,
          });
        }
      } else {
        toast.error("Container not found", {
          position: "bottom-right",
          autoClose: 5000,
          hideProgressBar: false,
          closeOnClick: true,
          pauseOnHover: true,
          draggable: true,
          progress: undefined,
        });
      }
    });
  };

  render() {
    const {
      files,
      audioUrls,
      selectedFileIndex,
      processing,
      progress,
      error,
      isPlaying,
      volume,
      zoom,
      chunks,
      lstLabels,
      mapChunkWithLabel,
      projectId,
    } = this.state;

    /*  height="100vh"
        width="100%"
        p={4}
        display="flex"
        flexDirection="column"
        justifyContent="center"
        alignItems="center"
        bg="gray.800"
        color="white"*/
    return (
      <Box>
        <Tabs
          isFitted
          variant="soft-rounded"
          colorScheme="green"
          minH="100vh"
          width="100%"
          bg="gray.800"
        >
          <TabList>
            <Tab>Label Manager</Tab>
            <Tab>Audio Labeling</Tab>
            <Tab>Chunk Converter</Tab>
            <Tab>AI Trainer</Tab>
          </TabList>
          <TabPanels h="95%">
            <TabPanel>
              <ObjLabelManager
                sendDataToParent={this.handleDataFromChildObjLabels}
                projectId={projectId}
              />
            </TabPanel>
            <TabPanel>
              <Input
                type="file"
                multiple
                onChange={this.handleFileChange}
                mt={4}
                accept="audio/*"
              />
              {processing && <Progress value={progress} mt={2} width="100%" />}
              {error && (
                <Text color="red.500" mt={2}>
                  {error}
                </Text>
              )}
              <VStack spacing={4} align="stretch" width="100%">
                <Wrap spacing={4} justify="center">
                  {files.map((file, index) => (
                    <WrapItem key={index}>
                      <Box
                        p={2}
                        borderWidth="1px"
                        borderRadius="lg"
                        cursor="pointer"
                        bg={
                          selectedFileIndex === index ? "teal.500" : "gray.700"
                        }
                        onClick={() => this.handleFileSelect(index)}
                      >
                        <Text>{file.name}</Text>
                      </Box>
                    </WrapItem>
                  ))}
                </Wrap>
                {selectedFileIndex !== null && audioUrls[selectedFileIndex] && (
                  <Box
                    p={4}
                    borderWidth="1px"
                    borderRadius="lg"
                    bg="gray.700"
                    width="100%"
                  >
                    <VStack
                      divider={<StackDivider borderColor="gray.200" />}
                      spacing={4}
                      align="stretch"
                    >
                      <Text mb={2}>Audio {selectedFileIndex + 1}</Text>
                      <div ref={this.waveSurferRef}></div>
                      <div ref={this.timelineRef}></div>
                      <HStack mt={2} spacing={4} width="100%">
                        <Button onClick={this.handlePlayPause}>
                          {isPlaying ? "Pause" : "Play"}
                        </Button>
                        <Text>Volume:</Text>
                        <Slider
                          value={volume}
                          min={0}
                          max={1}
                          step={0.1}
                          onChange={this.handleVolumeChange}
                          width="150px"
                        >
                          <SliderTrack>
                            <SliderFilledTrack />
                          </SliderTrack>
                          <SliderThumb />
                        </Slider>
                        <Text>Zoom:</Text>
                        <Slider
                          value={zoom}
                          min={0}
                          max={100}
                          onChange={this.handleZoomChange}
                          width="150px"
                        >
                          <SliderTrack>
                            <SliderFilledTrack />
                          </SliderTrack>
                          <SliderThumb />
                        </Slider>
                        <Button onClick={this.addRegion}>Add Region</Button>
                        <Button onClick={this.addMultipleRegions}>
                          Add Multiple Regions
                        </Button>
                      </HStack>
                      {chunks && (
                        <Box mt={4} h="40vh" overflow={"auto"}>
                          <Text mb={2}>Chunks:</Text>
                          <VStack align="stretch">
                            {chunks.map((chunk) => (
                              <Box
                                key={chunk.id}
                                p={2}
                                borderWidth="1px"
                                borderRadius="lg"
                              >
                                <Flex>
                                  <Text>
                                    Chunk {chunk.id}: {chunk.start}s -{" "}
                                    {chunk.end}s
                                  </Text>
                                  <Spacer />
                                  <ObjSelectLabel
                                    data={lstLabels}
                                    uniqPassIn={chunk}
                                    sendDataToParent={
                                      this.handleDataFromChildChunkWithLabel
                                    }
                                    uniqPassFile={
                                      this.state.files[
                                        this.state.selectedFileIndex
                                      ]
                                    }
                                  />
                                </Flex>
                              </Box>
                            ))}
                          </VStack>
                        </Box>
                      )}
                    </VStack>
                  </Box>
                )}
              </VStack>
            </TabPanel>

            <TabPanel>
              <ObjChunkConverter
                mapChunkWithLabel={mapChunkWithLabel}
                projectId={projectId}
              />
            </TabPanel>
            <TabPanel>
              <ObjAiTrainer projectId={projectId} />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Box>
    );
  }
}

export default withParams(ObjProject);
