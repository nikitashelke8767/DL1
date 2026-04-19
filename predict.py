from tensorflow.keras.models import load_model
import numpy as np
from src.utils import preprocess_face, emotion_labels

model = load_model("model/emotion_model.h5")

def predict_emotion(face):
    face = preprocess_face(face)
    prediction = model.predict(face)
    return emotion_labels[np.argmax(prediction)]