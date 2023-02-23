import os

import numpy as np
import tensorflow as tf
import glob

from tensorflow.python.data import AUTOTUNE


def prepare_images():
    print("------------prepare_images---------------")

    data_dir = os.path.join(os.getcwd(), 'flower_photos')
    print("data_dir", data_dir)
    roses = list(glob.glob("data_dir" + 'roses/*'))
    print(roses)
    # PIL.Image.open(str(roses[0]))

    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    ''' visualize the data '''
    import matplotlib.pyplot as plt

    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return [train_ds, val_ds]
