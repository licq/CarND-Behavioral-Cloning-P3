import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Lambda, Convolution2D, Flatten, Dense, Cropping2D, Dropout, ELU, BatchNormalization
from keras.models import Sequential
from keras.optimizers import RMSprop
from sklearn.utils import shuffle

tf.python.control_flow_ops = tf

input_shape = (160, 320, 3)


def image_path(path, full):
    last = full.split('/')[-1].strip()
    return os.path.join(path, 'IMG', last)


def read_driving_log(path, has_header=True):
    if has_header:
        df = pd.read_csv(os.path.join(path, 'driving_log.csv'))
    else:
        df = pd.read_csv(os.path.join(path, 'driving_log.csv'), header=None,
                         names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
    df['left'] = df.apply(lambda row: image_path(path, row['left']), axis=1)
    df['center'] = df.apply(lambda row: image_path(path, row['center']), axis=1)
    df['right'] = df.apply(lambda row: image_path(path, row['right']), axis=1)

    return df


def sample(df, limit, nb_small):
    small = df[np.abs(df['steering']) < limit]
    big = df[np.abs(df['steering']) >= limit]
    small_choice = np.random.choice(small.shape[0], nb_small)

    return shuffle(pd.concat([big, small.iloc[small_choice]]))


def flip_image(image, steering, random=True):
    if random and np.random.randint(2) > 0:
        return image, steering
    return np.fliplr(image), steering * -1.0


def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(((50, 20), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), init='he_normal'))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    model.add(Dense(1, init='he_normal'))

    print(model.summary())

    model.compile(loss='mse', optimizer="adam")
    return model


def openai_model():
    weight_init = 'he_normal'
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(((50, 20), (0, 0))))
    model.add(Convolution2D(3, 1, 1, subsample=(1, 1), border_mode='same', init=weight_init))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(16, 5, 5, subsample=(4, 4), border_mode='same', init=weight_init))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode='same', init=weight_init))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same', init=weight_init))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, init=weight_init))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1))

    print(model.summary())
    model.compile(loss='mse', optimizer='adam')

    return model


def new_model():
    weight_init = 'he_normal'
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(((50, 20), (0, 0))))
    model.add(BatchNormalization(mode=2, axis=1))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), init=weight_init))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), init=weight_init))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), init=weight_init))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, init=weight_init))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, init=weight_init))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, init=weight_init))
    model.add(ELU())
    model.add(Dense(50, init=weight_init))
    model.add(ELU())
    model.add(Dense(10, init=weight_init))
    model.add(ELU())
    model.add(Dense(1, init=weight_init))

    print(model.summary())

    model.compile(optimizer=RMSprop(0.0001), loss='mse')
    return model


def small_model():
    weight_init = 'glorot_normal'
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(((50, 20,), (0, 0))))
    model.add(BatchNormalization(mode=2, axis=1))
    model.add(Convolution2D(3, 3, 3, subsample=(2, 2), init=weight_init, activation='relu'))
    model.add(Convolution2D(9, 3, 3, subsample=(2, 2), init=weight_init, activation='relu'))
    model.add(Convolution2D(18, 3, 3, subsample=(2, 2), init=weight_init, activation='relu'))
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), init=weight_init, activation='relu'))
    model.add(Flatten())
    model.add(Dense(80, activation='relu', init=weight_init))
    model.add(Dense(15, activation='relu', init=weight_init))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='linear', init=weight_init))

    print(model.summary())
    model.compile(optimizer=RMSprop(0.00001), loss='mse')
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
            image = cv2.imread(image_name)
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
                    image = cv2.imread(image_name)
                    batch_images.append(image)
                    batch_steerings.append(steering)
                    if augment:
                        flipped_image, flipped_steering = flip_image(image, steering, random=False)
                        batch_images.append(flipped_image)
                        batch_steerings.append(flipped_steering)

            yield np.array(batch_images), np.array(batch_steerings)
