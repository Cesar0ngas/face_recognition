import os
from mtcnn import MTCNN
import cv2
import numpy as np
from numpy import expand_dims
from tensorflow.keras.models import load_model
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Cargar el modelo preentrenado de FaceNet
model = load_model('models/facenet_keras.h5')

# Inicializar el detector de rostros MTCNN
detector = MTCNN()

# Función para extraer el rostro de una imagen usando MTCNN
def extract_face(image_path, required_size=(160, 160)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)
    if results:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, required_size)
        return face
    else:
        return None

# Función para obtener el embedding de un rostro
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

# Directorio de las imágenes de entrenamiento
train_dir = 'C:/Users/cesco/Desktop/Personal/UPY/proyecto/dataset/train'
X_train, y_train = [], []

# Generar embeddings para todas las imágenes en el dataset
for person_name in os.listdir(train_dir):
    person_dir = os.path.join(train_dir, person_name)
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        face = extract_face(image_path)
        if face is not None:
            print(f"Rostro detectado para {person_name} en {image_name}")
            # Generar el embedding
            embedding = get_embedding(model, face)
            X_train.append(embedding)
            y_train.append(person_name)
        else:
            print(f"No se detectó rostro en {image_name} de {person_name}")

# Guardar los embeddings y las etiquetas
with open('models/embeddings.pkl', 'wb') as f:
    pickle.dump((X_train, y_train), f)

print("Embeddings generados y guardados correctamente.")
