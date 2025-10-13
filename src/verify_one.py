from model import model, predict_image

predicted_class, prediction = predict_image(model, '../resources/dataset/images/IMG_6984.jpg')

print(f"Classe prédite: {predicted_class}, Probabilités: {prediction}")