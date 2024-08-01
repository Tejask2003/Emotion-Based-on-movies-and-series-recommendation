from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import numpy as np
import base64
from fer import FER

app = Flask(__name__)

# Initialize the FER model
detector = FER(mtcnn=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    data = request.json
    image_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect emotions from the image
    emotion, score = detect_emotion_from_image(image)

    # Redirect based on the detected emotion
    if emotion == 'happy':
        return jsonify({'redirect': url_for('happy')})
    elif emotion == 'sad':
        return jsonify({'redirect': url_for('sad')})
    else:
        return jsonify({'redirect': url_for('neutral')})

@app.route('/happy')
def happy():
    return render_template('happy.html')

@app.route('/sad')
def sad():
    return render_template('sad.html')

@app.route('/neutral')
def neutral():
    return render_template('neutral.html')

def detect_emotion_from_image(image):
    # Detect emotions in the image
    result = detector.detect_emotions(image)
    if result:
        # Extract the emotion with the highest score
        top_emotion = max(result[0]['emotions'], key=result[0]['emotions'].get)
        score = result[0]['emotions'][top_emotion]
        return top_emotion, score
    return 'neutral', 0

if __name__ == '__main__':
    app.run(debug=True)
