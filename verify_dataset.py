# verify_dataset.py
import os
import matplotlib.pyplot as plt
import cv2
from newa_classes import NEWA_CHARACTERS

dataset_dir = "dataset_final/train"
class_counts = {}

for class_name in os.listdir(dataset_dir):
    count = len(os.listdir(
        os.path.join(dataset_dir, class_name)))
    class_counts[class_name] = count

# Check for imbalanced classes
min_count = min(class_counts.values())
max_count = max(class_counts.values())
avg_count = sum(class_counts.values()) / len(class_counts)

print(f"Total classes: {len(class_counts)}")
print(f"Total images:  {sum(class_counts.values())}")
print(f"Min per class: {min_count}")
print(f"Max per class: {max_count}")
print(f"Avg per class: {avg_count:.0f}")

# Flag classes with too few images
print("\n⚠ Classes with < 50 images (need more data):")
for cls, count in sorted(class_counts.items(),
                          key=lambda x: x[1]):
    if count < 50:
        print(f"  {cls}: {count} images")

# Plot distribution
plt.figure(figsize=(20, 5))
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=90, fontsize=7)
plt.title("Images per Class")
plt.tight_layout()
plt.savefig("dataset_distribution.png")
plt.show()