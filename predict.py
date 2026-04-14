import cv2
import numpy as np
from tensorflow.keras.models import load_model


model_buildings = load_model('model_buildings_50.keras')
model_roads = load_model('model_roads_50.keras')

def get_total_urbanization(image_path):
    # Зареждане и подготовка на снимката
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (256, 256))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)

    # Предвиждане от двата модела
    pred_b = model_buildings.predict(img_input)[0]
    pred_r = model_roads.predict(img_input)[0]

    #отделни маски
    mask_b = (pred_b > 0.3).astype(np.uint8) * 255
    mask_r = (pred_r > 0.3).astype(np.uint8) * 255


    combined_mask = cv2.bitwise_or(mask_b, mask_r)

    #kernel = np.ones((3, 3), np.uint8)
    #combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # Изчисляване на процентите
    perc_b = (np.sum(mask_b == 255) / (256 * 256)) * 100
    perc_r = (np.sum(mask_r == 255) / (256 * 256)) * 100
    total_perc = (np.sum(combined_mask == 255) / (256 * 256)) * 100

    print(f"Сгради: {perc_b:.2f}%")
    print(f"Пътища: {perc_r:.2f}%")
    print(f"ОБЩО ЗАСТРОЕНО: {total_perc:.2f}%")
    white_pixels = np.sum(combined_mask == 255)
    total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
    print(f"Бели пиксели: {white_pixels}")
    print(f"Общо пиксели: {total_pixels}")

    # Визуализация
    cv2.imshow('Buildings Only', mask_b)
    cv2.imshow('Roads Only', mask_r)
    cv2.imshow('Combined Result', combined_mask)
    cv2.waitKey(0)


# Тествай с твоя снимка
get_total_urbanization('F:/pycharm/gemini po slojen/test1.png')