"use client";


import React from "react";
import SimpleNAS from "./SimpleNAS";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
} from "chart.js";

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Title);

class ObjAiTrainer extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      activeTab: "trainer",
      files: [],
      currentFileIndex: 0,
      trainingProgress: 0,
      trainingComplete: false,
      error: null,
      isTraining: false,
      currentPage: 1,
      filesPerPage: 10,
      selectedFileContent: null,
      activeFileTab: "spectrogram",
      progressLogs: [],
      model: null,
      testData: "",
      testOutput: null,
    };
    this.nas = new SimpleNAS();
  }

  componentDidMount() {
    this.fetchFiles();
  }

  fetchFiles = async () => {
    const { projectId } = this.props;
    try {
      const response = await fetch(`/pback/files/${projectId}`);
      const data = await response.json();
      this.setState({ files: data.jsondata });
    } catch (error) {
      this.setState({ error: "Failed to fetch files." });
    }
  };

  startTraining = async () => {
    const { files } = this.state;
    if (files.length > 0) {
      this.setState(
        {
          isTraining: true,
          trainingProgress: 0,
          error: null,
          trainingComplete: false,
          currentFileIndex: 0,
          progressLogs: [],
        },
        this.trainNextFile
      );
    } else {
      this.setState({ error: "No files available for training." });
    }
  };

  trainNextFile = async () => {
    const { files, currentFileIndex } = this.state;
    if (currentFileIndex >= files.length) {
      this.setState({ trainingComplete: true, isTraining: false });
      return;
    }

    const currentFile = files[currentFileIndex];
    try {
      const dataFile = `/pback/files/view/${this.props.projectId}/jsondata/${currentFile}`;
      this.nas.dataFile = dataFile;

      await this.nas.loadData();
      this.nas.prepareTensors();
      await this.nas.trainModel();

      this.setState((prevState) => {
        const newLog = `Trained on file: ${currentFile}`;
        return {
          currentFileIndex: prevState.currentFileIndex + 1,
          trainingProgress:
            ((prevState.currentFileIndex + 1) / files.length) * 100,
          progressLogs: [newLog, ...prevState.progressLogs],
        };
      }, this.trainNextFile);
    } catch (error) {
      this.setState({
        error: `Failed to train on file: ${currentFile}`,
        isTraining: false,
      });
    }
  };

  handleClick = (event) => {
    this.setState({
      currentPage: Number(event.target.id),
      selectedFileContent: null,
    });
  };

  viewFileContent = async (filename) => {
    const { projectId } = this.props;
    try {
      const response = await fetch(
        `/pback/files/view/${projectId}/jsondata/${filename}`
      );
      if (!response.ok) {
        throw new Error(`Failed to load content of file: ${filename}`);
      }

      const contentType = response.headers.get("content-type");

      let data;
      if (contentType.includes("application/json")) {
        data = await response.json();
      } else {
        data = await response.text();
      }

      this.setState({ selectedFileContent: data });
    } catch (error) {
      this.setState({ error: `Failed to load content of file: ${filename}` });
    }
  };

  saveModel = async () => {
    await this.nas.saveModel();
  };

  loadModel = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const modelFile = new Blob([e.target.result], {
          type: "application/json",
        });
        const modelUrl = URL.createObjectURL(modelFile);
        await this.nas.importModel(modelUrl);
        this.setState({ model: this.nas.model });
      };
      reader.readAsText(file);
    }
  };

  handleTestInputChange = (event) => {
    this.setState({ testData: event.target.value });
  };

  testModel = async () => {
    const { testData } = this.state;
    try {
      const parsedData = JSON.parse(testData);
      if (!parsedData.spectrogram) {
        this.setState({ testOutput: "Invalid input data." });
        return;
      }
      const output = await this.nas.predict(parsedData);
      this.setState({ testOutput: output });
    } catch (error) {
      this.setState({ testOutput: "Invalid input data." });
    }
  };

  renderPageNumbers = () => {
    const { files, filesPerPage } = this.state;
    const pageNumbers = [];
    for (let i = 1; i <= Math.ceil(files.length / filesPerPage); i++) {
      pageNumbers.push(i);
    }

    return pageNumbers.map((number) => (
      <li
        key={number}
        id={number}
        onClick={this.handleClick}
        style={styles.pageNumber}
      >
        {number}
      </li>
    ));
  };

  render() {
    const {
      activeTab,
      files,
      trainingProgress,
      trainingComplete,
      error,
      isTraining,
      currentPage,
      filesPerPage,
      selectedFileContent,
      activeFileTab,
      progressLogs,
      testData,
      testOutput,
    } = this.state;

    const indexOfLastFile = currentPage * filesPerPage;
    const indexOfFirstFile = indexOfLastFile - filesPerPage;
    var currentFiles = null;

    try{
      currentFiles = files.slice(indexOfFirstFile, indexOfLastFile);
    }catch{
      console.log("No files?");
    }

    const spectrogramData = selectedFileContent?.spectrogram || [];
    const chartData = {
      labels: spectrogramData.map((_, index) => index),
      datasets: [
        {
          label: "Spectrogram",
          data: spectrogramData,
          fill: false,
          borderColor: "rgba(75,192,192,1)",
          tension: 0.1,
        },
      ],
    };

    return (
      <div style={styles.container}>
        <div style={styles.tabs}>
          <div
            style={{
              ...styles.tab,
              ...(activeTab === "trainer" ? styles.activeTab : {}),
            }}
            onClick={() => this.setState({ activeTab: "trainer" })}
          >
            AI Trainer
          </div>
          <div
            style={{
              ...styles.tab,
              ...(activeTab === "files" ? styles.activeTab : {}),
            }}
            onClick={() => this.setState({ activeTab: "files" })}
          >
            View Files
          </div>
          <div
            style={{
              ...styles.tab,
              ...(activeTab === "test" ? styles.activeTab : {}),
            }}
            onClick={() => this.setState({ activeTab: "test" })}
          >
            Test Model
          </div>
        </div>

        {activeTab === "trainer" && (
          <div style={styles.tabContent}>
            <h2 style={styles.header}>AI Model Trainer</h2>
            {error && <p style={styles.errorText}>{error}</p>}
            {trainingComplete ? (
              <p style={styles.successText}>
                Training complete! The model has been trained on all files.
              </p>
            ) : (
              <div style={styles.progressContainer}>
                <p style={styles.progressText}>
                  Training progress: {trainingProgress.toFixed(2)}%
                </p>
                <progress
                  style={styles.progressBar}
                  value={trainingProgress}
                  max="100"
                />
                {!isTraining && (
                  <button
                    style={styles.trainButton}
                    onClick={this.startTraining}
                    disabled={isTraining}
                  >
                    Start Training
                  </button>
                )}
              </div>
            )}
            <div style={styles.logContainer}>
              <h3 style={styles.subHeader}>Training Logs</h3>
              <ul style={styles.logList}>
                {progressLogs.map((log, index) => (
                  <li key={index} style={styles.logItem}>
                    {log}
                  </li>
                ))}
              </ul>
            </div>
            <button style={styles.saveButton} onClick={this.saveModel}>
              Save Model
            </button>
            <input
              type="file"
              onChange={this.loadModel}
              style={styles.uploadInput}
            />
          </div>
        )}

        {activeTab === "files" && (
          <div style={styles.tabContent}>
            <div style={styles.fileTabs}>
              <div
                style={{
                  ...styles.fileTab,
                  ...(activeFileTab === "spectrogram"
                    ? styles.activeFileTab
                    : {}),
                }}
                onClick={() => this.setState({ activeFileTab: "spectrogram" })}
              >
                Spectrogram
              </div>
              <div
                style={{
                  ...styles.fileTab,
                  ...(activeFileTab === "fileData" ? styles.activeFileTab : {}),
                }}
                onClick={() => this.setState({ activeFileTab: "fileData" })}
              >
                File Data
              </div>
            </div>

            <h3 style={styles.subHeader}>Files</h3>
            <ul style={styles.fileList}>
              {currentFiles.map((file, index) => (
                <li
                  key={index}
                  onClick={() => this.viewFileContent(file)}
                  style={styles.fileItem}
                >
                  {file}
                </li>
              ))}
            </ul>
            <ul style={styles.pagination}>{this.renderPageNumbers()}</ul>

            {selectedFileContent && activeFileTab === "spectrogram" && (
              <div>
                <h3 style={styles.subHeader}>Spectrogram Visualization</h3>
                <Line data={chartData} />
              </div>
            )}

            {selectedFileContent && activeFileTab === "fileData" && (
              <div>
                <h3 style={styles.subHeader}>File Content</h3>
                <pre style={styles.fileContent}>
                  {JSON.stringify(selectedFileContent, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}

        {activeTab === "test" && (
          <div style={styles.tabContent}>
            <h2 style={styles.header}>Test Model</h2>
            <textarea
              value={testData}
              onChange={this.handleTestInputChange}
              placeholder="Paste your test JSON data here"
              style={styles.testInput}
            />
            <button onClick={this.testModel} style={styles.testButton}>
              Test Model
            </button>
            {testOutput && (
              <div style={styles.testOutputContainer}>
                <h3 style={styles.subHeader}>Model Output</h3>
                <pre style={styles.testOutput}>
                  {JSON.stringify(testOutput, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    );
  }
}

const styles = {
  container: {
    width: "80%",
    margin: "0 auto",
    padding: "20px",
    borderRadius: "8px",
    backgroundColor: "transparent",
    textAlign: "center",
    color: "#f5f5f5",
  },
  tabs: {
    display: "flex",
    justifyContent: "center",
    marginBottom: "20px",
  },
  tab: {
    padding: "10px 20px",
    margin: "0 5px",
    cursor: "pointer",
    borderBottom: "2px solid transparent",
    transition: "border-bottom 0.3s",
  },
  activeTab: {
    borderBottom: "2px solid #007bff",
    color: "#007bff",
  },
  tabContent: {
    marginTop: "20px",
  },
  fileTabs: {
    display: "flex",
    justifyContent: "center",
    marginBottom: "20px",
  },
  fileTab: {
    padding: "10px 20px",
    margin: "0 5px",
    cursor: "pointer",
    borderBottom: "2px solid transparent",
    transition: "border-bottom 0.3s",
  },
  activeFileTab: {
    borderBottom: "2px solid #007bff",
    color: "#007bff",
  },
  header: {
    fontSize: "24px",
    marginBottom: "20px",
  },
  errorText: {
    color: "red",
    marginBottom: "20px",
    fontWeight: "bold",
  },
  successText: {
    color: "green",
    marginBottom: "20px",
    fontWeight: "bold",
  },
  progressContainer: {
    marginBottom: "20px",
  },
  progressText: {
    fontSize: "18px",
    marginBottom: "10px",
  },
  progressBar: {
    width: "100%",
    height: "20px",
    marginBottom: "20px",
    borderRadius: "8px",
    backgroundColor: "#e0e0e0",
  },
  trainButton: {
    padding: "10px 20px",
    fontSize: "16px",
    color: "#fff",
    backgroundColor: "#007bff",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    transition: "background-color 0.3s",
  },
  fileListContainer: {
    textAlign: "left",
    marginTop: "20px",
  },
  subHeader: {
    fontSize: "20px",
    marginBottom: "10px",
  },
  fileList: {
    listStyleType: "none",
    paddingLeft: "0",
    marginBottom: "20px",
  },
  fileItem: {
    padding: "10px",
    border: "1px solid #555",
    borderRadius: "4px",
    marginBottom: "10px",
    cursor: "pointer",
    backgroundColor: "#2c2c2c",
    transition: "background-color 0.3s",
  },
  pagination: {
    display: "flex",
    justifyContent: "center",
    listStyleType: "none",
    paddingLeft: "0",
    marginTop: "20px",
  },
  pageNumber: {
    padding: "8px 12px",
    margin: "0 5px",
    border: "1px solid #555",
    borderRadius: "4px",
    cursor: "pointer",
    backgroundColor: "#2c2c2c",
    transition: "background-color 0.3s",
  },
  fileContentContainer: {
    textAlign: "left",
    marginTop: "20px",
  },
  fileContent: {
    backgroundColor: "#1c1c1c",
    padding: "20px",
    borderRadius: "8px",
    color: "#f5f5f5",
    overflowX: "auto",
  },
  logContainer: {
    marginTop: "20px",
    textAlign: "left",
  },
  logList: {
    listStyleType: "none",
    paddingLeft: "0",
  },
  logItem: {
    padding: "10px",
    borderBottom: "1px solid #555",
  },
  saveButton: {
    padding: "10px 20px",
    fontSize: "16px",
    color: "#fff",
    backgroundColor: "#28a745",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    transition: "background-color 0.3s",
    marginTop: "20px",
  },
  uploadInput: {
    marginTop: "20px",
  },
  testInput: {
    width: "100%",
    height: "150px",
    padding: "10px",
    borderRadius: "8px",
    backgroundColor: "#2c2c2c",
    color: "#f5f5f5",
    border: "1px solid #555",
    marginBottom: "20px",
  },
  testButton: {
    padding: "10px 20px",
    fontSize: "16px",
    color: "#fff",
    backgroundColor: "#007bff",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    transition: "background-color 0.3s",
  },
  testOutputContainer: {
    marginTop: "20px",
    textAlign: "left",
  },
  testOutput: {
    backgroundColor: "#1c1c1c",
    padding: "20px",
    borderRadius: "8px",
    color: "#f5f5f5",
    overflowX: "auto",
  },
};

export default ObjAiTrainer;
