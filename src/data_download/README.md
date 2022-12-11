## Project Otways Dataset
This folder contains the code that give access to the Google Cloud Storage for the Project otways Dataset. Please follow the instructions below:

1. Install google cloud storage
pip install google-cloud-storage

2. Get the access key from Tony Zhai and save it to the same directory as this folder.

3. Open GoogleCloud_download.ipynb Bucket name is currently 'project_echo_dataset_1'. set dl_dir to the path where you want the dataset to be stored.

4. Make sure that the name of the key file XXXXXXXXXXXX.json is stored in the variable os.environ['GOOGLE_APPLICATION_CREDENTIALS']

5. Run the python script. It should take around 20 minutes depending on the speed of your connection.

### Other scripts

audio_cleaning is responsible for removing the silences and voices at the start of each audio clip
GoogleCloud_upload can upload datasets to the online storage bucket
web_scraping_ala is used to scrape audio data from Atlas of Living Australia website
txt_cleaning is used to extraxt species found in the Otways from the book: __Grant Palmer. Wildlife of the Otways and Shipwreck Coast. Clayton, VIC: CSIRO PUBLISHING, 2019.__
