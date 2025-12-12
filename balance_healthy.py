from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
from PIL import Image

src = "dataset/train/Healthy"
target_count = 2000  # equal to diseased
current = len(os.listdir(src))

datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

images = os.listdir(src)
i = 0

while len(os.listdir(src)) < target_count:
    img = load_img(os.path.join(src, images[i % current]))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    for batch in datagen.flow(x, batch_size=1, save_to_dir=src, save_prefix="aug", save_format="jpg"):
        break

    i += 1

print("✅ Healthy images balanced to", len(os.listdir(src)))
