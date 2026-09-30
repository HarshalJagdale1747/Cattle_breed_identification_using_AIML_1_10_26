from flask import Flask, send_from_directory, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# LOAD MODEL
model = tf.keras.models.load_model(
    "cattle_model.h5",
    compile=False,
    safe_mode=False
)

classes = ['Gir', 'Jersey', 'Sahiwal']

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/prediction.html')
def prediction():
    return send_from_directory('.', 'prediction.html')

@app.route('/contact.html')
def contact():
    return send_from_directory('.', 'contact.html')

@app.route('/image-prediction.html')
def image_prediction():
    return send_from_directory('.', 'image-prediction.html')

@app.route('/video-prediction.html')
def video_prediction():
    return send_from_directory('.', 'video-prediction.html')

@app.route('/camera-capture.html')
def camera_capture():
    return send_from_directory('.', 'camera-capture.html')

@app.route('/analysis.html')
def analysis():
    return send_from_directory('.', 'analysis.html')

@app.route('/information.html')
def information():
    return send_from_directory('.', 'information.html')

@app.route('/style.css')
def style():
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')

# PREDICTION
@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    image = Image.open(file).convert('RGB')
    image = image.resize((224, 224))

    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    result = classes[np.argmax(prediction)]

    return f"Predicted Breed: {result}"

# RUN SERVER
if __name__ == '__main__':
    app.run(debug=True)