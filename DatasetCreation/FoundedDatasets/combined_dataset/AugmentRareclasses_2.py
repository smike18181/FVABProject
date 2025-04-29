import os
import cv2
import random
import shutil
import numpy as np

classi_rare = [0]  # Inserisci qui le classi rare che vuoi aumentare

input_img_dir = "combined_dataset/images/train"
input_lbl_dir = "combined_dataset/labels/train"

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

    base_name = file.replace(".txt", "")

    # --- Flip orizzontale ---
    flipped_img = cv2.flip(image, 1)
    flipped_lbls = []
    for line in lines:
        cls, x, y, w, h = map(float, line.strip().split())
        x = 1.0 - x
        flipped_lbls.append(f"{int(cls)} {x} {y} {w} {h}")

    cv2.imwrite(os.path.join(input_img_dir, base_name + "_flip.jpg"), flipped_img)
    with open(os.path.join(input_lbl_dir, base_name + "_flip.txt"), "w") as f:
        f.writelines(l + "\n" for l in flipped_lbls)

    # --- Diverse luminosità ---
    brightness_factors = [0.6, 1.0, 1.4]  # più scura, normale, più chiara
    for i, factor in enumerate(brightness_factors):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite(os.path.join(input_img_dir, f"{base_name}_bright{i+1}.jpg"), bright_img)
        shutil.copy(label_path, os.path.join(input_lbl_dir, f"{base_name}_bright{i+1}.txt"))

    # --- Diverse sfocature ---
    blur_kernels = [(3, 3), (7, 7), (15, 15)]  # leggera, media, forte
    for i, ksize in enumerate(blur_kernels):
        blurred_img = cv2.GaussianBlur(image, ksize, 0)

        cv2.imwrite(os.path.join(input_img_dir, f"{base_name}_blur{i+1}.jpg"), blurred_img)
        shutil.copy(label_path, os.path.join(input_lbl_dir, f"{base_name}_blur{i+1}.txt"))
