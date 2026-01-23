import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("model_comparison_results.csv")
except FileNotFoundError:
    print("Error : model_comparison_results.csv not found.")
    exit()

plt.figure(figsize=(12, 8))
colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'] 

bars = plt.bar(df['Model'], df['Accuracy'], color=colors, alpha=0.7)

human_acc = 98.84
plt.axhline(y=human_acc, color='black', linestyle='--', linewidth=2, label=f'Human reference ({human_acc}%)')

resnet_sota = 99.20
plt.axhline(y=resnet_sota, color='red', linestyle=':', linewidth=2, label=f'Theoritical ResNet-50 ({resnet_sota}%)')


plt.ylabel('Accuracy Test (%)', fontsize=12)
plt.title('Performances comparison', fontsize=14)
plt.ylim(70, 100) 
plt.grid(axis='y', linestyle='--', alpha=0.3)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f}%", 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.legend(loc='lower left', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('final_comparison_with_baselines.png')
plt.show()

print("Graph saved : final_comparison_with_baselines.png")