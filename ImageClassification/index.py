import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

from utils import build_model

''' Load and prepare data '''

# Load Data from filesystem
data = tf.keras.utils.image_dataset_from_directory('data')

# convert data to iterator over numpy array
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# Scale Data
data = data.map(lambda x, y: (x / 255, y))
data.as_numpy_iterator().next()

# 5. Split Data into training, validation and test data
# - training data is used to train the model
# - validation data is used to validate the model training steps
# - test data should be kept independent of model creation
train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = int(len(data) * .1)

train_data = data.take(train_size)
validation_data = data.skip(train_size).take(val_size)
test_data = data.skip(train_size + val_size).take(test_size)

# Load existing or Build Deep Learning Model
if os.path.exists(os.path.join('models', 'imageclassifier.h5')):
    model = load_model(os.path.join('models', 'imageclassifier.h5'))
    # else create, compile,train and save
else:
    model = build_model(train_data, validation_data, test_data)

'''Predict'''

# img = cv2.imread('154006829.jpg') # good example
img = cv2.imread('8iAb9k4aT.jpg')  # bad example
plt.imshow(img)
plt.show()
# %%
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()
# %%
yhat = model.predict(np.expand_dims(resize / 255, 0))
# %%
yhat
# %%
if yhat > 0.5:
    print(f'Predicted class is Bad')
else:
    print(f'Predicted class is Good')
