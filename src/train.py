import os
import datetime
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mediapipe.framework.formats import image_format_pb2

#==================================================================#

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

#==================================================================#

def process_image_mediapipe(frame, target_size=(64, 64)):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        hand_roi = frame[y_min:y_max, x_min:x_max]
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, target_size)
        normalized = resized / 255.0
        return normalized
    return None

#==================================================================#

def process_image(frame, target_size=(64, 64)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    return normalized

#==================================================================#

def load_images_from_folder(folder, target_size=(64, 64)):
    images = []
    labels = []
    total_images = 0
    total_processed_images = 0

    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        print(f"Traitement du dossier : {label}")  # Debug
        nb_images_forlder = len(os.listdir(label_path))
        total_images += nb_images_forlder
        processed_images = 0
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"Image corrompue : {img_path}")  # Debug
                continue

            #processed_img = process_image_mediapipe(frame, target_size)
            processed_img = process_image(frame, target_size)
            if processed_img is not None:
                images.append(processed_img)
                labels.append(int(label))
                processed_images += 1
                total_processed_images += 1
                print(f"Images traitées : {processed_images}/{nb_images_forlder}")

    print(f"Traitement terminé. {total_processed_images}/{total_images} images traitées.")
    return np.array(images), np.array(labels)

#==================================================================#

X, y = load_images_from_folder('../resources/dataset/archive/fingers/train')
X = X.astype('float32')
X = np.expand_dims(X, axis=-1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#X_train = np.expand_dims(X_train, axis=-1)
#X_val = np.expand_dims(X_val, axis=-1)

#==================================================================#

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)
datagen.fit(X_train)

#==================================================================#

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1), padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.35),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#==================================================================#

checkpoint = ModelCheckpoint(
    '../resources/models/model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#==================================================================#

history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stopping, tensorboard_callback]
)