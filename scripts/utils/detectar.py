from mtcnn import MTCNN
import cv2
import numpy as np

def extract_face(image_path, required_size=(160, 160)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
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
