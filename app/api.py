from flask import Flask, request, jsonify
import cv2
import numpy as np
import sys
import os

# Agrega el directorio `scripts` a la ruta de búsqueda de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from face_recognition import recognize_person  # Importa desde scripts

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Prueba a llamar a la función y ver qué devuelve
    result = recognize_person(img)
    print("Resultado de recognize_person:", result)

    # Asigna los valores de forma segura para evitar el error
    if isinstance(result, tuple) and len(result) == 2:
        name, probability = result
        return jsonify({"name": name, "probability": probability})
    else:
        return jsonify({"error": "No face detected or unexpected output"}), 400
