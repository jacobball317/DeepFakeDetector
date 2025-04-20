import matplotlib.pyplot as plt
import numpy as np

# Define stages and corresponding scores
stages = [
    "Base Model",
    "Train Longer",
    "Deepen MLP",
    "Add BatchNorm",
    "Rollback + Normalize",
    "10 Frames/Video",
    "Learning Rate Decay",
    "Improve Face Detection",
    "Increase Frames per Video"
]

accuracy = [61.76, 64.42, 66.35, 64.42, 66.35, 70.83, 68.52, 81.25, 88.10]
precision = [65.96, 65.52, 66.67, 66.07, 67.24, 75.00, 70.40, 92.31, 92.68]
recall = [57.41, 69.09, 54.55, 67.27, 70.91, 70.59, 73.95, 70.59, 84.44]
f1_score = [61.39, 67.26, 60.00, 66.67, 69.03, 72.73, 72.13, 80.00, 88.37]

# Plot all metrics
plt.figure(figsize=(14, 8))
plt.plot(stages, accuracy, marker='o', label='Accuracy (%)')
plt.plot(stages, precision, marker='s', label='Precision (%)')
plt.plot(stages, recall, marker='^', label='Recall (%)')
plt.plot(stages, f1_score, marker='d', label='F1 Score (%)')

plt.title('Model Performance Metrics Across Upgrades')
plt.xlabel('Model Upgrade Stage')
plt.ylabel('Score (%)')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
