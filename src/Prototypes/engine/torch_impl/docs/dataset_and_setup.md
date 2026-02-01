[<- Back to Main Guide](model_integration_guide.md)

---

## Dataset and Setup Guide

This guide provides an overview of the Project Otways datasets hosted on Google Cloud Storage and instructions on how to access them.

### Project Otways Dataset

The dataset is available in three versions, stored in separate Google Cloud Storage buckets.

| Bucket Name           | Description                                                                                          |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| `project_echo_bucket_1` | Contains 3 audio files for each of the 118 labels, totalling 353 files (73 MB). Ideal for initial testing. |
| `project_echo_bucket_2` | Contains training data for the 118 species, totalling 7,161 files (168 MB). Overlaps 88% with bucket 3. |
| `project_echo_bucket_3` | Contains training clips for the 118 species, totalling 7,536 files (349 MB).                          |

The Google Cloud Project name is `sit-23t1-project-echo-25288b9`.

### Accessing the Dataset

Please follow these steps to set up your environment and download the data.

1.  **Install Google Cloud Storage Library**
    ```bash
    pip install google-cloud-storage
    ```

2.  **Install the Google Cloud CLI**
    -   Please follow the official instructions to [install the Google Cloud SDK](https://cloud.google.com/sdk/docs/install) for your operating system.

3.  **Authenticate Your Account**
    -   Once the CLI is installed, open a new terminal and run the following command. It will open a web browser for you to log in.
    ```bash
    gcloud auth application-default login
    ```
    -   Please log in using your Deakin credentials (e.g., `xxxxxx@deakin.edu.au`) and complete the two-factor authentication. A successful authentication will be confirmed in your terminal.

4.  **Download the Data**
    -   The `GoogleCloud_download.ipynb` notebook is used to download files from the storage buckets.
    -   Before running the notebook, please open it and set the `dl_dir` variable to the local directory path where you wish to store the dataset.
    -   The `bucket_name` variable is pre-set to `project_echo_bucket_1` but can be changed to any of the buckets listed above.
    -   Execute the notebook cells to begin the download.

### Other Utility Scripts

The `data/` directory contains several other scripts for data management and processing:

-   `audio_cleaning`: Removes silences and human voices from the start of audio clips.
-   `audio_cleaning_2`: A refined version of the cleaning script that uses sound onset detection to ensure no clip is pure silence.
-   `GoogleCloud_upload.ipynb`: A notebook for uploading datasets to a Google Cloud Storage bucket.
-   `web_scraping_ala`: Scrapes audio data from the Atlas of Living Australia website.
-   `txt_cleaning`: Used to extract species names found in the Otways from the book "Wildlife of the Otways and Shipwreck Coast".

---
[<- Back to Main Guide](model_integration_guide.md)
