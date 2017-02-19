import csv
from keras.layers import Flatten, Cropping2D, Lambda, Convolution2D, MaxPooling2D, Activation, Dropout, Dense
from sklearn.utils import shuffle
import cv2
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

samples = []
with open('data/driving_log.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def argument(file_name, angle):
    name = 'data/' + file_name.strip()
    image = cv2.imread(name)
    flipped_image = cv2.flip(image, 0)
    flipped_angle = -1.0 * angle
    return [image, flipped_image], [angle, flipped_angle]


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_angle = float(batch_sample[3])
                batch_images, batch_angles = argument(batch_sample[0], center_angle)
                batch_images_left, batch_angles_left = argument(batch_sample[1], center_angle + 0.2)
                batch_images_right, batch_angles_right = argument(batch_sample[2], center_angle - 0.2)
                images.extend(batch_images + batch_images_left + batch_images_right)
                angles.extend(batch_angles + batch_angles_left + batch_angles_right)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5))
model.add(MaxPooling2D(strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5))
model.add(MaxPooling2D(strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5))
model.add(MaxPooling2D(strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile('adam', 'mse')
history_object = model.fit_generator(generator=train_generator, samples_per_epoch=len(train_samples) * 6,
                                     validation_data=validation_generator, nb_val_samples=len(validation_samples) * 6,
                                     nb_epoch=1)

model.save('model.h5')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
