import os
from collections import Counter

label_dir = "combined_dataset/labels/train"
class_counts = Counter()

for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue
    with open(os.path.join(label_dir, file), "r") as f:
        for line in f:
            cls_id = int(line.strip().split()[0])
            class_counts[cls_id] += 1

print("Distribuzione delle classi:")
for cls, count in class_counts.items():
    print(f"Classe {cls}: {count} istanze")
