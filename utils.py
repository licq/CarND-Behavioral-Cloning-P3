import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Lambda, Convolution2D, Flatten, Dense, Cropping2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.utils import shuffle

tf.python.control_flow_ops = tf

data_path = 'data/'


def sample(df, limit, nb_small):
    small = df[np.abs(df['steering']) < limit]
    big = df[np.abs(df['steering']) >= limit]
    small_choice = np.random.choice(small.shape[0], nb_small)

    return shuffle(pd.concat([big, small.iloc[small_choice]]))


def flip_image(image, steering, random=True):
    if random and np.random.randint(2) > 0:
        return image, steering
    return np.fliplr(image), steering * -1.0


def read_image(name):
    return cv2.imread(data_path + name.strip())


def preprocess_image(image):
    # image = image[51:141, :, :]
    # return cv2.resize(image, (200, 66))
    return image


def nvidia_model(input_shape, with_cropping=True):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    if with_cropping:
        model.add(Cropping2D(((50, 20), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    print(model.summary())

    model.compile(loss='mse', optimizer=Adam())
    return model


def random_data_generator(df, batch_size, augment=True):
    cameras = ['left', 'center', 'right']
    corrections = [0.25, 0, -0.25]
    while True:
        batch = df.sample(batch_size)
        batch_images = []
        batch_steerings = []
        for _, row in batch.iterrows():
            cam_index = np.random.randint(3)
            image_name = row[cameras[cam_index]]
            steering = row['steering'] + corrections[cam_index]
            image = preprocess_image(read_image(image_name))
            if augment:
                image, steering = flip_image(image, steering)
            batch_images.append(image)
            batch_steerings.append(steering)

        yield np.array(batch_images), np.array(batch_steerings)


def data_generator(df, batch_size, augment=True):
    cameras = ['left', 'center', 'right']
    corrections = [0.25, 0, -0.25]
    while True:
        shuffle(df)
        for offset in range(0, len(df), batch_size):
            batch = df.iloc[offset: offset + batch_size]
            batch_images = []
            batch_steerings = []
            for _, row in batch.iterrows():
                for camera, correction in zip(cameras, corrections):
                    image_name = row[camera]
                    steering = row['steering'] + correction
                    image = preprocess_image(read_image(image_name))
                    batch_images.append(image)
                    batch_steerings.append(steering)
                    if augment:
                        flipped_image, flipped_steering = flip_image(image, steering, random=False)
                        batch_images.append(flipped_image)
                        batch_steerings.append(flipped_steering)

            yield np.array(batch_images), np.array(batch_steerings)
