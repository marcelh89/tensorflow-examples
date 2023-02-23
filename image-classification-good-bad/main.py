from model import train_model
from prepare import prepare_images
from predict import predict

training_dataset, valitation_dataset = prepare_images()
model = train_model(training_dataset, valitation_dataset)
predict(model)
