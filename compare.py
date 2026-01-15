import matplotlib.pyplot as plt
import numpy as np

models = ['Custom CNN', 'LeNet-5', 'VGG-16', 'ResNet-50', 'Humain (Ref)']
accuracy_scores = [95.72, 91.20, 98.50, 99.20, 98.84]
inference_speed = [5, 3, 25, 40] 

plt.figure(figsize=(10, 6))
colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'gold']

bars = plt.bar(models, accuracy_scores, color=colors)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval}%", ha='center', va='bottom', fontweight='bold')

plt.ylim(85, 101) 
plt.ylabel('Accuracy %')
plt.title('Models comparaison (GTSRB)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()