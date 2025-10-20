import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from model import model

#==================================================================#

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

#==================================================================#

# Fonction de prétraitement
def preprocess_frame(frame):
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Redimensionner
    resized = cv2.resize(gray, (64, 64))
    # Normaliser
    normalized = resized / 255.0
    # Ajouter une dimension pour le batch
    input_tensor = tf.expand_dims(normalized, 0)
    return input_tensor

#==================================================================#

# Fonction pour détecter la main (version simple avec seuillage)
def detect_hand(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Seuil pour isoler la main (à ajuster selon ton environnement)
    _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    # Trouver les contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Prendre le plus grand contour (supposé être la main)
        hand_contour = max(contours, key=cv2.contourArea)
        # Recadrer autour de la main
        x, y, w, h = cv2.boundingRect(hand_contour)
        hand_roi = frame[y:y+h, x:x+w]
        return hand_roi
    return None

#==================================================================#

def detect_hand_mediapipe(frame):
    # Convertir la frame pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        return frame
    return None

    if results.multi_hand_landmarks:
        # On prend la première main détectée (car max_num_hands=1)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Récupérer les coordonnées normalisées des landmarks
        h, w, _ = frame.shape
        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

        # Calculer la bounding box de la main
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Ajouter une marge pour inclure toute la main
        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        # Extraire la ROI
        hand_roi = frame[y_min:y_max, x_min:x_max]
        return hand_roi
    return None

#==================================================================#

# Ouverture de la caméra
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Détecter la main
    hand_roi = detect_hand_mediapipe(frame)
    if hand_roi is not None:
        # Prétraiter la ROI
        input_tensor = preprocess_frame(hand_roi)
        # Inférence
        pred = model.predict(input_tensor)
        classe = np.argmax(pred, axis=1)[0]
        print(f"Nombre de doigts levés : {classe}")  # Affichage dans le terminal

        cv2.putText(frame, f"{classe} doigts", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Aucune main detecte", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Reconnaissance', frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
