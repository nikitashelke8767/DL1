import cv2
import numpy as np

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def preprocess_face(face):
    face = cv2.resize(face, (48,48))
    face = face / 255.0
    face = np.reshape(face, (1,48,48,1))
    return face