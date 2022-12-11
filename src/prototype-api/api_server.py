import sys

from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import datetime

sys.path.append('../baseline')
import baseline/echo_tfimm_model

# target classes to display to user
target_classes = ['nightjar', 'skylark', 'yellow-faced honeyeater', 'feral goat', 
                  'sambar deer', 'grey shrikethrush', 'australian raven', 'fallow deer', 
                  'yellow robin', 'cat', 'whistler', 'white-plumed honeyeater', 
                  'brown rat', 'pied currawong', 'wild pig']





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
         
         # TODO run the classifier here
         species='Cat'
      else:
         species = "No File Uploaded!"
      
      return render_template('index.html', species=species)

if __name__ == '__main__':
    print("starting app")
    app.run(debug = True)