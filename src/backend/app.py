from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import base64
from MRI_system.script import try_YOLOv8
from MRI_system.script import show_slices
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def numpy_to_base64(image_array):
    """Convert a numpy array to a base64 string."""
    image = Image.fromarray(np.uint8(image_array * 255))  # Convert array to an image
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify(message="Hello from Flask!")

@app.route('/api/upload', methods=['POST'])
def upload_image():
    # Check if there is a file in the request
    if 'image' not in request.files:
        return jsonify(error="No file part"), 400
    
    model = request.form.get('subject')
    file = request.files['image']
    
    # Check the file extension
    if not file.filename.endswith('.nii'):
        return jsonify(error="Only .nii files are allowed"), 400
    
    # Save the file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Check if CUDA is available
    import torch
    print("CUDA is available:" + str(torch.cuda.is_available()))

    # Get slices
    axial_slice, sagittal_slice, coronal_slice = show_slices(os.path.join(os.getcwd(), file_path))

    # Convert slices to base64 strings
    axial_base64 = numpy_to_base64(axial_slice)
    sagittal_base64 = numpy_to_base64(sagittal_slice)
    coronal_base64 = numpy_to_base64(coronal_slice)

    # Return the images as base64 strings in the JSON response
    return jsonify(
        uploaded=True,
        images=[axial_base64, sagittal_base64, coronal_base64, axial_base64]
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
