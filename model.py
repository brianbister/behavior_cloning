import csv
import json
import os

from keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
from sklearn.model_selection import train_test_split

input_shape = (160, 320, 3,)

# Set up the network
model = Sequential()

# Normalize the input
model.add(BatchNormalization(input_shape=input_shape))

# Convolutions
model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2, 2)))
model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2, 2)))
model.add(Convolution2D(64, 3, 3, border_mode="same"))

# Fully connected layer
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(216))
model.add(Dense(1))

model.compile(loss="mean_squared_error",
              optimizer=Adam(lr=0.00001),
              metrics=["acc"])

images = []
steering_angles = []

with open("driving_log.csv", "rt") as drive_log:
	csv_reader = csv.reader(drive_log)
	for row in csv_reader:
		images.append(image.img_to_array(image.load_img("IMG/" + os.path.basename(row[0]))))  # Center
		images.append(image.img_to_array(image.load_img("IMG/" + os.path.basename(row[1]))))  # Left
		images.append(image.img_to_array(image.load_img("IMG/" + os.path.basename(row[2]))))  # Right
		steering_angles.append(float(row[3]))
		steering_angles.append(float(row[3]))
		steering_angles.append(float(row[3]))

images = np.array(images)
steering_angles = np.array(steering_angles)

# Split training and test sets
images_train, images_val, steering_angles_train, steering_angles_val = train_test_split(images, steering_angles, test_size=0.3)
generator = image.ImageDataGenerator()

nb_epoch = 1
samples_per_epoch = len(images_train)

# Train the model
model.fit_generator(generator.flow(images_train, steering_angles_train),
					samples_per_epoch=samples_per_epoch,
					nb_epoch=nb_epoch,
					validation_data=(images_val, steering_angles_val))
loss = model.evaluate(images_val, steering_angles_val)
print("Loss: {}".format(loss))

# Save the model
with open("model.json", "w+") as model_file:
          json.dump(model.to_json(), model_file)

model.save_weights("model.h5")
