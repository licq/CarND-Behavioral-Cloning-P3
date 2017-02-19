import numpy as np
import csv
from sklearn.model_selection import train_test_split

from scipy import misc
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Activation, Flatten, Lambda, Cropping2D, MaxPooling2D, Dense

DRIVING_LOG_CSV = 'data/driving_log.csv'
IMAGE_PATH = 'data/'
CORRECTION = 0.2


def process_image(img):
    return img


car_images = []
steering_angles = []

with open(DRIVING_LOG_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        steering_center = float(row['steering'])
        steering_left = steering_center + CORRECTION
        steering_right = steering_center - CORRECTION

        img_center = process_image(np.asarray(misc.imread(IMAGE_PATH + row['center'].strip())))
        img_left = process_image(np.asarray(misc.imread(IMAGE_PATH + row['left'].strip())))
        img_right = process_image(np.asarray(misc.imread(IMAGE_PATH + row['center'].strip())))

        car_images.extend((img_center, img_left, img_right))
        steering_angles.extend((steering_center, steering_left, steering_right))

X = np.array(car_images)
y = np.array(steering_angles)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5))
model.add(MaxPooling2D(strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5))
model.add(MaxPooling2D(strides=(2, 2)))
model.add(Activation('relu'))
# model.add(Convolution2D(48, 5, 5))
# model.add(MaxPooling2D(strides=(2, 2)))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile('adam', 'mse', metrics=['accuracy'])
model.fit(X, y, nb_epoch=5, validation_split=0.2)
model.save('model.h5')
