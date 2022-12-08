from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

app = Flask(__name__)

@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def echo_upload_file():
   if request.method == 'POST':
      print(request)
      f = request.files['file']
      filename = secure_filename(f.filename)
      f.save(filename)
      # TODO run the classifier here
      return 'file uploaded successfully'
		
if __name__ == '__main__':
    print("starting app")
    app.run(debug = True)