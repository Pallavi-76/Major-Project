from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load models
cnn_model = tf.keras.models.load_model('./models/parkinson_disease_detectioncnn.h5')
densenet_model = tf.keras.models.load_model('./models/parkinson_disease_detection_densenet201.h5')

def preprocess_image(image_path, model_type):
    img = cv2.imread(image_path)
    
    if model_type == 'spiral':  # CNN model (expects grayscale)
        img_size = (128, 128)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=-1) 
    elif model_type == 'wave': 
        img_size = (224, 224)
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
    else:
        raise ValueError("Invalid model type")
    
    img = np.expand_dims(img, axis=0)
    return img

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the selected option and file
        image_type = request.form['image_type']
        uploaded_file = request.files['image']
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join('./static/uploads', filename)
        uploaded_file.save(file_path)

        # Preprocess the image
        try:
            img = preprocess_image(file_path, image_type)
        except ValueError as e:
            return jsonify({'error': str(e)})

        # Make predictions based on the selected option
        if image_type == 'spiral':
            model = cnn_model
        elif image_type == 'wave':
            model = densenet_model
        else:
            return jsonify({'error': 'Invalid selection'})

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction[0])
        labels = ['Healthy', 'Parkinson']
        result = labels[predicted_class]

        # Return result
        return render_template('result.html', result=result, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
