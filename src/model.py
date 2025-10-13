import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2
import numpy as np

#==================================================================#

def preprocess_image(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

#==================================================================#

def predict_image(model, image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return predicted_class, prediction

#==================================================================#

def predict_image_from_directory(model, directory):
    total_results = 0
    correct_results = 0
    for directory_path in os.listdir(directory):
        dir_full_path = os.path.join(directory, directory_path)
        for image_name in os.listdir(dir_full_path):
            image_path = os.path.join(dir_full_path, image_name)
            predicted_class, prediction = predict_image(model, image_path)
            print(f"Image: {image_name}, Classe prédite: {predicted_class}, Probabilités: {prediction}")
            total_results += 1
            if str(predicted_class) == directory_path:
                correct_results += 1
    accuracy = (correct_results / total_results) * 100 if total_results > 0 else 0
    print(f"Précision totale: {accuracy:.2f}% ({correct_results}/{total_results})")
            

model = tf.keras.models.load_model('../resources/models/model.h5')