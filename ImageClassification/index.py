import os

from keras.models import load_model

from utils import build_model, prepare_data, predict_multiple

model_name = 'imageclassifier.h5'
model_path = os.path.join('models', model_name)

# Load existing or Build Deep Learning Model
if os.path.exists(model_path):
    model = load_model(model_path)
    # else create, compile,train and save
else:
    # Load and prepare train
    train_data, validation_data, test_data = prepare_data()
    print(train_data)
    model = build_model(train_data, validation_data, test_data, model_path)

'''Predict'''

# single image
# predict_single(model)

# multiple images
predict_multiple(model)
