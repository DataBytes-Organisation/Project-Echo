
### Prototype API server and front end

This demostration API assumes the input files are 5 second clips in the flac format.  As this is a prototype more sophisticated methods such as loading larger files and find the most likely clip within the file was not implemented here.  If there are any changes to the input processing of the model, these need to be reflected in the echo_audio_pipeline.py file

To run the api server:

- Open a miniconda terminal

- Activate the dev environment: > conda activate dev

- Navigate to Project-Echo\src\prototype_api\

- run> python api_server.py

To run the client you need a web browser.

- navigate to http://127.0.0.1:5000/

