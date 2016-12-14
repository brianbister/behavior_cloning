import csv
import json
import os

import cv2
from keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
from sklearn.model_selection import train_test_split

input_shape = (32, 64, 3,)

# Set up the network
model = Sequential()

# Normalize the input
model.add(BatchNormalization(input_shape=input_shape))

# Convolutions
model.add(Convolution2D(16, 5, 5, border_mode="same", subsample=(2, 2), activation="relu"))
model.add(Convolution2D(32, 5, 5, border_mode="same", subsample=(2, 2), activation="relu"))

# Fully connected layer
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(loss="mean_squared_error",
              optimizer=Adam(lr=0.0001),
              metrics=["acc"])

images = []
steering_angles = []

with open("driving_log.csv", "rt") as drive_log:
	csv_reader = csv.reader(drive_log)
	for row in csv_reader:
		image_data = image.img_to_array(image.load_img("IMG/" + os.path.basename(row[0])))
		images.append(image_data) # Center
		images.append(cv2.flip(image_data, 1))
		steering_angles.append(float(row[3]))
		steering_angles.append(-float(row[3]))

images = np.array(images)
images = np.array([cv2.resize(image_data, (input_shape[1], input_shape[0])) for image_data in images])
steering_angles = np.array(steering_angles)

permutation = np.random.permutation(len(images))
images, steering_angles = images[permutation], steering_angles[permutation]

# Split training and test sets
images_train, images_val_and_test, steering_angles_train, steering_angles_val_and_test = train_test_split(images, steering_angles, test_size=0.3)
images_val, images_test, steering_angles_val, steering_angles_test = train_test_split(images_val_and_test, steering_angles_val_and_test, test_size=0.5)


generator = image.ImageDataGenerator()

nb_epoch = 5
samples_per_epoch = len(images_train)

# Train the model
model.fit_generator(generator.flow(images_train, steering_angles_train),
					samples_per_epoch=samples_per_epoch,
					nb_epoch=nb_epoch,
					validation_data=(images_val, steering_angles_val))
loss = model.evaluate(images_test, steering_angles_test)
print("Loss: {}".format(loss))

# Save the model
with open("model.json", "w+") as model_file:
          json.dump(model.to_json(), model_file)

model.save_weights("model.h5")
