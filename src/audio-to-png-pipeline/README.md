-----------------------------------------------------------
For most of these libraries to work - Python 3.9 is required 
On mac, open terminal in this directory and execute the following command:
https://www.gyan.dev/ffmpeg/builds/
```
pip install apache-beam
```
```
brew install libav
```
```
brew install ffmpeg
```
```
pip install -r env_dev.txt
```

This will install all the necessary libraries

windows: https://windowsloop.com/install-ffmpeg-windows-10/

Please make sure that if you are on a Mac, that your path resebles the below example path
if you are on windows, please ensure that your path looks like this:

WIN
self.DATASET_PATH = 'C:\\Users\\steph\\Documents\\birdclef2022\\'
MAC
#self.DATASET_PATH  = '/Users/stephankokkas/Downloads/birdclef2022/'

make sure this is in the same format for either window or mac
#WIN
self._SET_OUTPUT_DIR = 'C:\\Users\\steph\\Downloads\\'
#MAC
#self._SET_OUTPUT_DIR = '/Users/stephankokkas/Downloads/'