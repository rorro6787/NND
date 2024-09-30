# app.py
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from MRI_system.script import try_YOLOv8

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello from Flask!")

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.json
    return jsonify(received=data)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    # Check if there is a file in the request
    if 'image' not in request.files:
        return jsonify(error="No file part"), 400
    
    file = request.files['image']
    
    # If the user does not select a file, the browser may submit an empty file
    if file.filename == '':
        return jsonify(error="No selected file"), 400
    
    # Save the file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    image_res_path = try_YOLOv8(os.path.join(os.getcwd(), file_path))
    # Return the saved file
    return send_file(os.path.join(image_res_path, filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
