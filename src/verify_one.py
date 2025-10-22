from model import model, predict_image

predicted_class, prediction = predict_image(model, '../resources/dataset/my_images/1/IMG_6984.jpg')
print(f"Classe prédite: {predicted_class}, Probabilités: {prediction}")