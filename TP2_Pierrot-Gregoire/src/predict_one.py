from model import, predict_image
from create_model import load_model

model = load_model()
predicted_class, prediction = predict_image(model, '../resources/dataset/my_images/1/IMG_6984.jpg')
print(f"Classe prédite: {predicted_class}, Probabilités: {prediction}")