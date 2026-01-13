import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import cv2

DATA_DIR = "./1" 
IMG_SIZE = 32
NUM_CLASSES = 43

def load_data(data_dir):
    images = []
    labels = []
    for class_id in range(NUM_CLASSES):
        path = os.path.join(data_dir, 'Train', str(class_id))
        for img_name in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, img_name))
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(class_id)
            except Exception as e:
                print(f"Error on  {img_name}: {e}")
                
    return np.array(images), np.array(labels)

X, y = load_data(DATA_DIR)

# image size normalization (0-1)
X = X.astype('float32') / 255.0

# Train (80%) / Test (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Dropout(0.2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_val, y_val)
)

model.save('traffic_sign_model.h5')
print("Model saved as 'traffic_sign_model.h5'")