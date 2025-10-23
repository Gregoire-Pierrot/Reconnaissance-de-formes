import cv2
import math
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
from create_model import load_model, save_model

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Pour éviter le spam des logs GPU

import tensorflow as tf
tf.config.run_functions_eagerly(True)


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
        #margin = 40
        #x_min = max(0, x_min - margin)
        #y_min = max(0, y_min - margin)
        #x_max = min(w, x_max + margin)
        #y_max = min(h, y_max + margin)
        margin_percent = 0.2
        margin_x = int((x_max - x_min) * margin_percent)
        margin_y = int((y_max - y_min) * margin_percent)
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)

        # Extraire la ROI
        #hand_roi = frame[y_min:y_max, x_min:x_max]
        #return hand_roi
        return x_min, y_min, x_max, y_max
    return None

#==================================================================#

def gettwtonepoints(frame, hands_detector=hands):
    """
    Retourne les 21 landmarks d'une main si détectée.

    Args:
        frame: image BGR (numpy array) provenant de la webcam.
        hands_detector: instance de mp.solutions.hands.Hands (optionnel).

    Returns:
        dict ou None:
        {
            'landmarks_norm': [(x, y, z), ...]   # 21 tuples normalisés (0..1)
            'landmarks_px'  : [(x_px, y_px, z), ...]  # 21 tuples en pixels (int,int, float z)
            'handedness'    : 'Left' or 'Right' or None,
            'mp_landmarks'  : MediaPipe landmarks object (utile si tu veux dessiner)
        }
        ou None si aucune main détectée.
    """
    if frame is None:
        return None

    # Convertir en RGB pour MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    # Prendre la première main détectée
    hand_landmarks = results.multi_hand_landmarks[0]

    h = frame.shape[0]
    w = frame.shape[1]
    landmarks_norm = []
    landmarks_px = []

    for lm in hand_landmarks.landmark:
        landmarks_norm.append((lm.x, lm.y, lm.z))
        landmarks_px.append((int(lm.x * w), int(lm.y * h), lm.z))

    # Récupérer handedness si présent (facultatif)
    handedness = None
    if results.multi_handedness:
        try:
            handedness = results.multi_handedness[0].classification[0].label
        except Exception:
            handedness = None

    return {
        'landmarks_norm': landmarks_norm,
        'landmarks_px': landmarks_px,
        'handedness': handedness,
        'mp_landmarks': hand_landmarks
    }

#==================================================================#

def count_finger_thumb(lm1, lm2, lm3, base_coords):
    if lm1[0] > base_coords[0] & lm1[1] > base_coords[1]:
        if lm2[0] > lm1[0] & lm2[1] > lm1[1]:
            if lm3[0] > lm2[0] & lm3[1] > lm2[1]:
                return 1
    if lm1[0] > base_coords[0] & lm1[1] < base_coords[1]:
        if lm2[0] > lm1[0] & lm2[1] < lm1[1]:
            if lm3[0] > lm2[0] & lm3[1] < lm2[1]:
                return 1
    if lm1[0] < base_coords[0] & lm1[1] < base_coords[1]:
        if lm2[0] < lm1[0] & lm2[1] < lm1[1]:
            if lm3[0] < lm2[0] & lm3[1] < lm2[1]:
                return 1
    if lm1[0] < base_coords[0] & lm1[1] > base_coords[1]:
        if lm2[0] < lm1[0] & lm2[1] > lm1[1]:
            if lm3[0] < lm2[0] & lm3[1] > lm2[1]:
                return 1
    return 0

#==================================================================#

def count_finger(lm1, lm2, lm3, base_coords):
    distance1 = math.sqrt((lm1[0] - base_coords[0])**2 + (lm1[1] - base_coords[1])**2)
    distance2 = math.sqrt((lm2[0] - base_coords[0])**2 + (lm2[1] - base_coords[1])**2)
    distance3 = math.sqrt((lm3[0] - base_coords[0])**2 + (lm3[1] - base_coords[1])**2)
    if distance3 < distance2 or distance3 < distance1:
        return 0
    return 1

#==================================================================#

def count_fingers(landmarks):
    counter = 0
    base_coords_x, base_coords_y, _ = landmarks[0]
    base_coords = (base_coords_x, base_coords_y)
    #20->18, 16->14, 12->10, 8->6, 4->2
    lm1_x, lm1_y, _ = landmarks[2]
    lm2_x, lm2_y, _ = landmarks[3]
    lm3_x, lm3_y, _ = landmarks[4]
    #counter += count_finger_thumb((lm1_x, lm1_y), (lm2_x, lm2_y), (lm3_x, lm3_y), base_coords)
    i = -2
    while i <= 14:
        i = i + 4
        lm1_x, lm1_y, _ = landmarks[i]
        lm2_x, lm2_y, _ = landmarks[i+1]
        lm3_x, lm3_y, _ = landmarks[i+2]
        counter += count_finger((lm1_x, lm1_y), (lm2_x, lm2_y), (lm3_x, lm3_y), base_coords)
    return counter

#==================================================================#

target_size = (64, 64)
batch_size = 128
sample_size = 32
train_buffer = deque(maxlen=batch_size)
iteration = 0
cap = cv2.VideoCapture(0)
model = load_model()

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    if not ret:
        break
    
    show_frame = frame
    hand_roi = detect_hand_mediapipe(show_frame)
    if hand_roi is not None:
        x_min, y_min, x_max, y_max = hand_roi
        hand_roi = show_frame[y_min:y_max, x_min:x_max]
        
        points = gettwtonepoints(show_frame)
        counter = 0
        if points is not None:
            landmarks_px = points['landmarks_px']
            counter = count_fingers(landmarks_px)
            #print(f"Nombre de doigts levés : {counter}")
            for i, (x, y, z) in enumerate(landmarks_px):
                if x_min < x < x_max and y_min < y < y_max:
                    cv2.circle(show_frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(show_frame, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        inp = preprocess_frame(hand_roi)
        train_buffer.append((inp, counter))
        print(len(train_buffer))

        if len(train_buffer) >= batch_size:
            batch_indices = np.random.choice(len(train_buffer), sample_size, replace=False)
            batch = [train_buffer[i] for i in batch_indices]
            X_batch = np.array([x.numpy().squeeze() for x, _ in batch])
            X_batch = np.expand_dims(X_batch, -1)
            y_batch = np.array([y for _, y in batch], dtype=np.float32)

            print("X_batch shape:", X_batch.shape)
            print("y_batch shape:", y_batch.shape)

            history = model.fit(X_batch, y_batch, epochs=1, verbose=0)
            for i in sorted(batch_indices, reverse=True):
                del train_buffer[i]

            iteration += 1
            print(f"[TRAIN] Iter {iteration}")
            
            if iteration % 5 == 0:  # par exemple, toutes les 5 itérations
                save_model(model)
                print("[SAVE] Modèle sauvegardé !")

        pred = model.predict(inp, verbose=0)
        pred_label = int(np.argmax(pred))

        cv2.putText(hand_roi, f"Algo:{counter}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        cv2.putText(hand_roi, f"IA:{pred_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
        
        hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        hand_roi = cv2.resize(hand_roi, (h, w))   

    else:
        cv2.putText(frame, "Aucune main detectee", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        hand_roi = frame
        
    cv2.imshow('Image', hand_roi)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        save_model(model)
        print("[SAVE] Modèle sauvegardé !")
        break

cap.release()
cv2.destroyAllWindows()