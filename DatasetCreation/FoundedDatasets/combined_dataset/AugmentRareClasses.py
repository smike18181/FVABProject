import os
import cv2
import random
import shutil
import numpy as np

classi_rare = [52]

#modifica il path in base a quale combined_dataset vuoi usare, val o train.
input_img_dir = "combined_dataset/images/val"
input_lbl_dir = "combined_dataset/labels/val"

for file in os.listdir(input_lbl_dir):
    if not file.endswith(".txt"):
        continue

    label_path = os.path.join(input_lbl_dir, file)
    img_path = os.path.join(input_img_dir, file.replace(".txt", ".jpg"))

    with open(label_path, "r") as f:
        lines = f.readlines()

    if not any(int(l.split()[0]) in classi_rare for l in lines):
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Errore nel leggere: {img_path}")
        continue

    # --- Augmentazione 1: Luminosità (HSV) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness_factor = random.uniform(0.6, 1.4)  # più scura o più chiara
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    new_img_name = file.replace(".txt", "_bright.jpg")
    new_lbl_name = file.replace(".txt", "_bright.txt")

    cv2.imwrite(os.path.join(input_img_dir, new_img_name), bright_img)
    shutil.copy(label_path, os.path.join(input_lbl_dir, new_lbl_name))

    # --- Augmentazione 2: Sfocatura (Gaussian Blur) ---
    blurred_img = cv2.GaussianBlur(image, (7, 7), 0)

    new_img_name = file.replace(".txt", "_blur.jpg")
    new_lbl_name = file.replace(".txt", "_blur.txt")

    cv2.imwrite(os.path.join(input_img_dir, new_img_name), blurred_img)
    shutil.copy(label_path, os.path.join(input_lbl_dir, new_lbl_name))
