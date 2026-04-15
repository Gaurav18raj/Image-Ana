# src/preprocess.py

import os
import numpy as np
from PIL import Image   # faster than cv2

IMG_SIZE = 32   # keep small for speed

def load_data(data_path):
    data = []
    labels = []

    print("Checking path:", data_path)

    for category in ["cats", "dogs"]:
        path = os.path.join(data_path, category)
        label = 0 if category == "cats" else 1

        print("Reading folder:", path)

        if not os.path.exists(path):
            print("❌ Folder not found:", path)
            continue

        files = os.listdir(path)
        print(f"Total images in {category}: {len(files)}")

        # LIMIT for testing (IMPORTANT)
        for i, img in enumerate(files[:100]):   # only 100 images

            if i % 20 == 0:
                print(f"{category}: Processing {i} images")

            try:
                img_path = os.path.join(path, img)

                image = Image.open(img_path).convert("RGB")
                image = image.resize((IMG_SIZE, IMG_SIZE))

                image = np.array(image).flatten()

                data.append(image)
                labels.append(label)

            except Exception as e:
                print("Skipping:", img_path)

    print("Finished loading data!")
    return np.array(data), np.array(labels)