from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration for image uploads
app.config["IMAGE_UPLOADS"] = os.path.join(app.root_path, 'static/Images')  # Ensure this path is correct

@app.route("/home", methods=['POST', 'GET'])
def upload_image():
    if request.method == 'POST':
        image = request.files.get('file')  # Get the uploaded file

        if not image or image.filename == '':
            print("Image must have a file name")
            return redirect(request.url)

        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

        return render_template("main.html", filename=filename)

    return render_template("main.html")

@app.route("/gallery")
def gallery():
    images = os.listdir(app.config["IMAGE_UPLOADS"])
    return render_template("gallery.html", images=images)

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/")
def home():
    return redirect(url_for('upload_image'))

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename="Images/" + filename), code=301)

if __name__ == "__main__":
    app.run(port=5000)
