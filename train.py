import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import RepeatedStratifiedKFold

DATA_DIR = "./1" 
IMG_SIZE = 32
NUM_CLASSES = 43
EPOCHS = 15
BATCH_SIZE = 64

def load_data(data_dir):
    images = []
    labels = []
    print("Loading dataset...")
    for class_id in range(NUM_CLASSES):
        path = os.path.join(data_dir, 'Train', str(class_id))
        if not os.path.exists(path): continue
        for img_name in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, img_name))
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(class_id)
            except Exception:
                pass 
    return np.array(images), np.array(labels)

def create_my_cnn():
    return models.Sequential([
        layers.Input(shape=(32, 32, 3)), 
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(43, activation='softmax')
    ])

def create_lenet5():
    return models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(6, (5, 5), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(43, activation='softmax')
    ])

def create_vgg_small():
    return models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(43, activation='softmax')
    ])

def create_resnet50():
    base_model = tf.keras.applications.ResNet50(
        include_top=False, 
        weights=None,
        input_shape=(32, 32, 3)
    )
    
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(43, activation='softmax')
    ])
    return model

#Normalization
X, y = load_data(DATA_DIR)
X = X.astype('float32') / 255.0

#Cross-validation 5x2
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)

model_factories = {
    #"Custom_CNN": create_my_cnn,
    #"LeNet5": create_lenet5,
    #"VGG_Small": create_vgg_small
    "ResNet50": create_resnet50
}

final_stats = {}

for name, factory in model_factories.items():
    print(f"\n>>> Starting 5x2 CV for: {name}")
    fold_accs = []
    
    for i, (train_idx, val_idx) in enumerate(rskf.split(X, y)):
        model = factory()
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        
        model.fit(X[train_idx], y[train_idx], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        
        _, acc = model.evaluate(X[val_idx], y[val_idx], verbose=0)
        fold_accs.append(acc)
        print(f"  Fold {i+1}/10: {acc*100:.2f}%")
    
    mean_val = np.mean(fold_accs) * 100
    std_val = np.std(fold_accs) * 100
    final_stats[name] = (mean_val, std_val)
    
    print(f"DONE: {name} | Accuracy: {mean_val:.2f}% (+/- {std_val:.2f}%)")
    
    #Saving last model to test
    model.save(f"{name}_final.h5")

print("\n--- ALL MODELS TRAINED ---")
for name, (m, s) in final_stats.items():
    print(f"{name}: {m:.2f}% ± {s:.2f}%")