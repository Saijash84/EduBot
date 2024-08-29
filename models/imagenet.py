# imagenet_model.py
from tensorflow.keras.models import load_model

model = load_model('imagenet_model.h5')

def predict_function(image_data):
    # Preprocess image_data as needed
    return model.predict(image_data)
