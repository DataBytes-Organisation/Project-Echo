# AI Data Storage and Model Repository

This project is a Flask-based backend server designed to store training data and AI models. It provides endpoints for uploading spectrogram data, AI models, and running inference using the stored models.

## Features

- **Upload Spectrogram Data**: Allows users to upload spectrogram data along with labels and project IDs. The data is stored in JSON format.
- **Upload AI Models**: Supports uploading AI models in JSON and H5 formats. TensorFlow.js models can be converted to TensorFlow models.
- **List Files**: Lists all uploaded files for a given project ID.
- **View Files**: Allows users to view specific files within a project.
- **Run Inference**: Runs inference using a specified AI model and provided data.
- **Health Check**: A simple endpoint to check if the server is running.

## Endpoints

### Upload Spectrogram Data

- **URL**: `/upload/spectrogram`
- **Method**: `POST`
- **Description**: Uploads spectrogram data along with labels and project ID.
- **Request Parameters**:
  - `project_id` (form data): The ID of the project.
  - `labels` (form data): The labels for the spectrogram data in JSON format.
  - `spectrogram` (form data): The spectrogram data in JSON format.
- **Response**: JSON object with a message and MD5 hash of the spectrogram data.

### Upload AI Model

- **URL**: `/upload/aiModel`
- **Method**: `POST`
- **Description**: Uploads an AI model file for a given project ID.
- **Request Parameters**:
  - `project_id` (form data): The ID of the project.
  - `file` (form data): The AI model file (JSON or H5 format).
- **Response**: JSON object with a message and the path where the file is saved.

### List Files

- **URL**: `/files/<project_id>`
- **Method**: `GET`
- **Description**: Lists all uploaded files for a given project ID.
- **Response**: JSON object with the list of files.

### View File

- **URL**: `/files/view/<project_id>/<folder>/<filename>`
- **Method**: `GET`
- **Description**: Allows users to view a specific file within a project.
- **Response**: The requested file.

### Run Inference

- **URL**: `/models/<project_id>/<model_name>`
- **Method**: `POST`
- **Description**: Runs inference using a specified AI model and provided data.
- **Request Parameters**:
  - `data` (JSON): The data to be used for inference.
- **Response**: JSON object with the predictions.

### Health Check

- **URL**: `/`
- **Method**: `GET`
- **Description**: A simple endpoint to check if the server is running.
- **Response**: A message indicating the server is running.

```
pip install -r requirements.txt
```