import os
import tensorflow as tf
from tensorflow.keras import layers, models

#==================================================================#

def build_model():
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
    
    return model

#==================================================================#

def create_model():
    model = build_model()
    model.save('../resources/models/model.h5')
    return model
    
#==================================================================#

def save_model(model):
    model.save('../resources/models/model.h5')

#==================================================================#
    
def load_model():
    if os.path.exists('../resources/models/model.h5'):
        model = tf.keras.models.load_model('../resources/models/model.h5')
        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        return model
    else:
        return create_model()
    
#==================================================================#