from model import model, predict_image

predicted_class, prediction = predict_image(model, '../resources/dataset/images/WIN_20251013_20_11_48_Pro.jpg')
print(f"Classe prédite: {predicted_class}, Probabilités: {prediction}")