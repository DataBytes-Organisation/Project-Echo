## Using the pipeline:

For this pipeline to work - it is imperative you have the correct libraries installed. Tensorflow, tensorflow-io and ffmpeg are the most important libraries to have installed on your computer and working!

For a tutorial to install ffmpeg:
https://windowsloop.com/install-ffmpeg-windows-10/
https://www.gyan.dev/ffmpeg/builds/

For most of these libraries to work - Python 3.9 is required - so, on mac and in a terminal run:
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

You have the ability in cell number 3 : Preprocessing Pipeline to define an output path and a dataset path. This is very important you get this correct otherwise - again - the pipeline will not work. 
If you are on a mac, ensure your dataset path has single '/' between each dir and a trailing '/' as shown in the example below. For windows, ensure thatt you have double '\\' in your path
and that you too have a trailing '\\'. If there are any issues, please make sure you create an issue in GitHub so it is recorded and corrected. 

Thanks

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


## Output
The final pipeline will load the tensors into three dataframes - train, test, and validation - with the label of that tensor. Alternatively, you can view the raw tensors in your output directory. 

The train, test, and validation directories contain raw mp3 files and are not tensors. For the tensors, they will be in the tensors directory. 