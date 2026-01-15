import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


DATA_DIR = "./1" 
IMG_SIZE = 32

model = load_model('traffic_sign_model.h5')

test_df = pd.read_csv(os.path.join(DATA_DIR, 'Test.csv'))
y_test = test_df['ClassId'].values
test_images_paths = test_df['Path'].values

X_test = []

for img_path in test_images_paths:
    full_path = os.path.join(DATA_DIR, img_path)
    
    img = cv2.imread(full_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X_test.append(img)

X_test = np.array(X_test).astype('float32') / 255.0

loss, accuracy = model.evaluate(X_test, y_test)

print(f"\n--- FINAL RESULTS ---")
print(f"Accuracy on the batch test  : {accuracy * 100:.2f}%")

idx = random.randint(0, len(X_test) - 1)
sample_img = X_test[idx]
true_label = y_test[idx]

prediction = model.predict(np.expand_dims(sample_img, axis=0))
predicted_class = np.argmax(prediction)

print(f"Real class : {true_label}")
print(f"Predict class : {predicted_class}")

plt.imshow(cv2.cvtColor((sample_img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title(f"Real: {true_label} | Predictions: {predicted_class}")
plt.show()

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(20, 15))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.xlabel('Predicted Classes')
plt.ylabel('Real Classes')
plt.title('Condusion matrix - GTSRB')
plt.show()

print("\nDetailed classification report :")
print(classification_report(y_test, y_pred))