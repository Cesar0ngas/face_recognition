import os
import requests
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import pickle

# Definir la ruta del modelo y la URL en Google Cloud Storage
model_path = 'models/facenet_keras.h5'
model_url = 'https://storage.googleapis.com/facenet_keras/facenet_keras.h5'  # Cambia esto con la URL de tu archivo en Google Cloud Storage

# Función para descargar el modelo
def download_model():
    print("Descargando el modelo desde Google Cloud Storage...")
    response = requests.get(model_url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1 MB chunks
                f.write(chunk)
        print("Modelo descargado con éxito.")
    else:
        raise Exception("Error al descargar el modelo, verifica la URL y el acceso público.")

# Verificar si el archivo existe y tiene un tamaño razonable, si no, descargarlo
if not os.path.exists(model_path) or os.path.getsize(model_path) < 1_000_000:  # Tamaño mínimo de 1MB como ejemplo
    download_model()

# Intentar cargar el modelo, si falla, volver a descargar e intentar cargar de nuevo
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error al cargar el modelo: {e}. Volviendo a descargar el archivo...")
    os.remove(model_path)  # Eliminar el archivo corrupto
    download_model()  # Volver a descargar
    model = load_model(model_path)  # Intentar cargar nuevamente

# Cargar el clasificador y el codificador
with open('models/svm_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

detector = MTCNN()

# Función para obtener el embedding de un rostro
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# Función para reconocer a la persona en un frame de video o imagen
def recognize_person(frame):
    results = detector.detect_faces(frame)
    if results:  # Si hay al menos un rostro detectado
        for result in results:
            x1, y1, width, height = result['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            
            # Obtener el embedding y realizar la predicción
            embedding = get_embedding(face)
            yhat_class = classifier.predict([embedding])[0]
            yhat_prob = classifier.predict_proba([embedding])[0]

            # Verificar la probabilidad y el nombre predicho
            try:
                class_probability = yhat_prob[yhat_class] * 100 if yhat_class < len(yhat_prob) else 0.0
                predicted_name = encoder.inverse_transform([yhat_class])[0]
                return predicted_name, class_probability
            except (IndexError, ValueError) as e:
                print(f"Error en la predicción: {e}")
                return "Desconocido", 0.0
    else:
        print("No se detectó ningún rostro")
        return "Desconocido", 0.0
