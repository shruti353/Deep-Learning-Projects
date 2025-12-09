"""
Flask web application for Handwritten Digit Recognition.
"""

import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2
import base64
import io

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.utils import ModelUtils

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'artifacts/models/best_model.h5'
)

try:
    model = ModelUtils.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image_data, target_size=(28, 28)):
    """
    Preprocess uploaded image.
    
    Args:
        image_data: Base64 encoded image or PIL Image
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image array
    """
    try:
        # If base64 string, decode it
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Remove header
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
        else:
            # Assume it's already a PIL Image
            image = image_data.convert('L')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize
        img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Invert colors if needed (white digits on black background)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalize
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        img_array = img_array.reshape(1, target_size[0], target_size[1], 1)
        
        return img_array
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

@app.route('/')
def home():
    """Render home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image
        image = Image.open(file.stream)
        
        # Preprocess
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        
        # Prepare response
        probabilities = [float(p) for p in predictions]
        
        response = {
            'success': True,
            'prediction': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/draw', methods=['POST'])
def predict_drawing():
    """Handle predictions from canvas drawings."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get base64 image data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
        
        image_data = data['image']
        
        # Preprocess
        processed_image = preprocess_image(image_data)
        
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]
        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        
        # Prepare response
        probabilities = [float(p) for p in predictions]
        
        response = {
            'success': True,
            'prediction': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)