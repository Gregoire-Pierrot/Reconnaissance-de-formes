import cv2
import numpy as np
import os
from pathlib import Path

# ---------- Fonctions de transformation ----------

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def incline_image(image, direction='up'):
    h, w = image.shape[:2]
    delta = 0.2 * w  # déformation de 20%
    if direction == 'up':
        src = np.float32([[0, h], [w, h], [0, 0]])
        dst = np.float32([[0, h], [w, h], [delta, 0]])
    elif direction == 'down':
        src = np.float32([[0, 0], [w, 0], [0, h]])
        dst = np.float32([[0, 0], [w, 0], [delta, h]])
    elif direction == 'left':
        src = np.float32([[w, 0], [w, h], [0, 0]])
        dst = np.float32([[w, 0], [w - delta, h], [0, 0]])
    elif direction == 'right':
        src = np.float32([[0, 0], [0, h], [w, 0]])
        dst = np.float32([[0 + delta, 0], [0, h], [w, 0]])
    else:
        return image

    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

# ---------- Fonction principale d'augmentation ----------

def augment_image(image_path, output_path, base_name):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[!] Erreur lors du chargement de {image_path}")
        return
    
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    directions = ['up', 'down', 'left', 'right']
    variant_id = 0

    for angle in angles:
        rotated = rotate_image(image, angle)

        # Variantes : originale, bruitée, floutée
        variants = {
            "orig": rotated,
            "noise": add_noise(rotated),
            "blur": blur_image(rotated),
        }

        for vname, variant in variants.items():
            # Image de base (sans inclinaison)
            filename = f"{base_name}_a{angle}_{vname}_{variant_id:02}.jpg"
            cv2.imwrite(str(output_path / filename), variant)
            variant_id += 1

            # Inclinaisons
            for dir in directions:
                inclined = incline_image(variant, dir)
                filename = f"{base_name}_a{angle}_{vname}_{dir}_{variant_id:02}.jpg"
                cv2.imwrite(str(output_path / filename), inclined)
                variant_id += 1

# ---------- Fonction pour parcourir le dataset ----------

def process_dataset(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for digit_dir in input_dir.iterdir():
        if not digit_dir.is_dir():
            continue

        label = digit_dir.name
        output_label_dir = output_dir / label
        output_label_dir.mkdir(parents=True, exist_ok=True)

        for img_file in digit_dir.glob("*.*"):
            base_name = img_file.stem
            augment_image(img_file, output_label_dir, base_name)
            print(f"[✓] Augmenté: {img_file}")

# ---------- Lancement ----------

INPUT_DIR = "../resources/dataset/sign-language-digits-datasets-master/Dataset"
OUTPUT_DIR = "../resources/dataset/augmented_dataset/"
process_dataset(INPUT_DIR, OUTPUT_DIR)
