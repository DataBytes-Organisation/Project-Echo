import path
import sys
 
# directory reach
directory = path.Path(__file__).abspath()
 
# setting path
sys.path.append(directory.parent)
sys.path.append(directory.parent.parent)

# import prototype model
from prototype_models import echo_tfimm_model
import echo_audio_pipeline

import tensorflow as tf
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename

# target classes to display to user
target_classes = ['nightjar', 'skylark', 'yellow-faced honeyeater', 'feral goat', 
                  'sambar deer', 'grey shrikethrush', 'australian raven', 'fallow deer', 
                  'yellow robin', 'cat', 'whistler', 'white-plumed honeyeater', 
                  'brown rat', 'pied currawong', 'wild pig']

# audio pipeline
pipeline = echo_audio_pipeline.EchoAudioPipeline()

# audio classifier 
classifier = echo_tfimm_model.EchoTfimmModel()
classifier.build([None, 
                  echo_tfimm_model.MODEL_INPUT_IMAGE_HEIGHT, 
                  echo_tfimm_model.MODEL_INPUT_IMAGE_WIDTH, 
                  echo_tfimm_model.MODEL_INPUT_IMAGE_CHANNELS])
classifier.load_weights('../prototype_models/models/baseline_timm_model.hdf5')

# this runs the pipeline and prediction model and returns target and probability
def predict_class(filename):
   melspec = pipeline.convert_audio_to_melspec(filename)
   #print(melspec.shape)
   result = classifier.predict(melspec)
   #print(result, result.shape, tf.nn.softmax(result))
   target_index = tf.argmax(tf.squeeze(result)).numpy()
   target_class = target_classes[target_index]    
   target_proba = 100.0*tf.nn.softmax(result)[0,target_index].numpy()
   target_proba = str(round(target_proba, 2))
   #print("target class: ", target_class)  
   return target_class, target_proba

# run the API server
app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')

@app.route('/')
def demo_page():
   return render_template('index.html', species='')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def echo_upload_file():
   if request.method == 'POST':
      print(request)
      f = request.files['file']
      filename = secure_filename(f.filename)
      if len(filename) > 0:
         f.save(filename)
         prediction = predict_class(filename)
         species = "File: " + filename + "    Species: [" + prediction[0] + "]     Probability: [" + str(prediction[1]+"%]")
      else:
         species = "No File Uploaded!"
      
      return render_template('index.html', species=species)

if __name__ == '__main__':
    print("starting app")
    app.run(debug = True)