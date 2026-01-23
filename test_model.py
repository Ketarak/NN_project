import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import csv
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

DATA_DIR = "./1" 
IMG_SIZE = 32
MODELS_TO_EVALUATE = ["Custom_CNN_final.h5", "LeNet5_final.h5", "VGG_Small_final.h5", "ResNet50_final.h5"]
CSV_OUTPUT = "model_comparison_results.csv"

test_df = pd.read_csv(os.path.join(DATA_DIR, 'Test.csv'))
y_test = test_df['ClassId'].values
test_images_paths = test_df['Path'].values

X_test = []
print(f"Loading and preprocessing {len(test_images_paths)} images")
for img_path in test_images_paths:
    full_path = os.path.join(DATA_DIR, img_path)
    img = cv2.imread(full_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X_test.append(img)

X_test = np.array(X_test).astype('float32') / 255.0

results_for_csv = []
print("\n" + "="*70)
print(f"{'Model':<20} | {'Acc (%)':<10} | {'Loss':<10} | {'Speed (ms)':<10}")
print("-" * 70)

for model_file in MODELS_TO_EVALUATE:
    if os.path.exists(model_file):
        model = load_model(model_file, compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        start_time = time.time()
        model.predict(X_test[:100], verbose=0)
        end_time = time.time()
        inf_speed = ((end_time - start_time) / 100) * 1000 
        
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        acc_percentage = accuracy * 100
        
        print(f"{model_file:<20} | {acc_percentage:>8.2f}% | {loss:>8.4f} | {inf_speed:>8.2f}")
        
        results_for_csv.append({
            "Model": model_file, 
            "Accuracy": acc_percentage,
            "Loss": loss,
            "Inference_Speed": inf_speed
        })
        
        if "Custom" in model_file:
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(20, 15))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
            plt.title(f'Confusion matrix - {model_file}')
            plt.xlabel('Predicted class')
            plt.ylabel('Real class')
            plt.savefig('confusion_matrix_final.png')
            plt.close() 
            print(f"\nClassification Report ({model_file}) :\n")
            print(classification_report(y_test, y_pred))
    else:
        print(f"{model_file:<20} | File not found")

print("="*70)

keys = results_for_csv[0].keys()
with open(CSV_OUTPUT, 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
    dict_writer.writeheader()
    dict_writer.writerows(results_for_csv)

print(f"\nResults saved in : {CSV_OUTPUT}")