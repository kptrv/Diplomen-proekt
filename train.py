import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from unet_model import build_unet


def load_data(img_dir, mask_dir, size=(256, 256)):
    images = []
    masks = []

    # само .png
    filenames = [f for f in os.listdir(img_dir) if f.lower().endswith('.png')]
    print(f"Намерени .png снимки: {len(filenames)}")

    for filename in filenames:
        img_path = os.path.join(img_dir, filename)

        mask_path = os.path.join(mask_dir, filename)

        if os.path.exists(mask_path):
            # Зареждане на снимката
            img = cv2.imread(img_path)
            img = cv2.resize(img, size)

            # Зареждане на маската
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, size)

            images.append(img)
            masks.append(mask)
        else:
            print(f"Внимание: Не намерих маска за снимка {filename}")

    # Превръщане в numpy масиви и нормализация
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(masks, dtype=np.float32) / 255.0
    y = np.expand_dims(y, axis=-1)

    return X, y



X, y = load_data('dataset_roads/images', 'dataset_roads/masks')

if len(X) > 0:
    print(f"Започвам обучение с {len(X)} двойки снимка-маска...")
    model = build_unet()
    print(f"Общо заредени изображения (X): {X.shape[0]}")
    print(f"Общо заредени маски (y): {y.shape[0]}")
    model.fit(X, y, epochs=100, batch_size=4)
    model.save('model_roads.keras')
else:
    print("Грешка: Не бяха заредени никакви данни. Провери имената на папките!")