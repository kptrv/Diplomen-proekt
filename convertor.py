import os
from PIL import Image


def batch_convert(input_dir, output_dir, mode="RGB"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))]

    for filename in files:
        img_path = os.path.join(input_dir, filename)
        with Image.open(img_path) as img:
            # Важно: Конвертираме в правилния режим
            # "RGB" за снимки, "L" за черно-бели маски
            converted_img = img.convert(mode)

            new_name = os.path.splitext(filename)[0] + ".png"
            converted_img.save(os.path.join(output_dir, new_name), "PNG")

    print(f"Готово! Конвертирани {len(files)} файла в {mode} режим.")


# --- ИЗПОЛЗВАНЕ ---
# 1. Конвертирай снимките (Цветни)
batch_convert('F:/pycharm/gemini po slojen/dataset_roads/images', 'F:/pycharm/gemini po slojen/datasettt/images', mode="RGB")

# 2. Конвертирай маските (Черно-бели)
batch_convert('F:/pycharm/gemini po slojen/dataset_roads/masks', 'F:/pycharm/gemini po slojen/datasettt/masks', mode="L")