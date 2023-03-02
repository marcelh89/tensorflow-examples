import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


def prepare_data():
    # Load Data from filesystem
    data = tf.keras.utils.image_dataset_from_directory(os.path.join('data', 'train'))

    # convert train to iterator over numpy array
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()

    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])

    # Scale Data
    data = data.map(lambda x, y: (x / 255, y))
    data.as_numpy_iterator().next()

    # 5. Split Data into training, validation and test train
    # - training train is used to train the model
    # - validation train is used to validate the model training steps
    # - test train should be kept independent of model creation
    train_size = int(len(data) * .7)
    val_size = int(len(data) * .2)
    test_size = int(len(data) * .1)

    train_data = data.take(train_size)
    validation_data = data.skip(train_size).take(val_size)
    test_data = data.skip(train_size + val_size).take(test_size)

    return [train_data, validation_data, test_data]


def build_model(train, val, test, model_path):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    # logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

    # 8. Plot Performance
    # %%
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
    # %%
    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
    # %% md
    # 9. Evaluate
    # %%

    # %%
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    # %%
    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    # %%
    print(pre.result(), re.result(), acc.result())

    # save model
    model.save(model_path, save_format='h5')

    return model


def good_or_bad(value):
    # %%
    if value > 0.5:
        return 'good'
    else:
        return 'bad'


def predict_single(model):
    # img = cv2.imread(os.path.join('test', 'good.jpeg'))  # good example
    img = cv2.imread(os.path.join('data/test', 'bad.jpeg'))  # bad example

    plt.imshow(img)
    plt.show()
    # %%
    resize = tf.image.resize(img, (256, 256))
    plt.imshow(resize.numpy().astype(int))
    plt.show()
    # %%
    yhat = model.predict(np.expand_dims(resize / 255, 0))


def predict_multiple(model):
    test_images = os.listdir("data/test")
    print(test_images)

    fig, ax = plt.subplots(ncols=2, figsize=(20, 20))
    for idx, img_src in enumerate(test_images):
        img = cv2.imread(os.path.join('data/test', img_src))
        resize = tf.image.resize(img, (256, 256))
        ax[idx].imshow(resize.numpy().astype(int))
        prediction = model.predict(np.expand_dims(resize / 255, 0))
        ax[idx].title.set_text(img_src + "- " + str(prediction) + " - " + good_or_bad(prediction))
        # ax[idx].suptitle(str(prediction))

    plt.show()
