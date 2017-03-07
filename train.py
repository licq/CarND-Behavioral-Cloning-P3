import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Lambda, Convolution2D, Flatten, Dense, Cropping2D, Dropout, ELU, MaxPooling2D
from keras.models import Sequential
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

tf.python.control_flow_ops = tf

MODEL_FILE = 'model.h5'
EPOCHS = 10
BATCH_SIZE = 64
INPUT_SHAPE = (160, 320, 3)


def image_path(path, full):
    last = full.split('/')[-1].strip()
    return os.path.join(path, 'IMG', last)


def read_driving_log(path, has_header=False):
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


def naive_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=INPUT_SHAPE))
    model.add(Flatten())
    model.add(Dense(1))

    return model


def lenet_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=INPUT_SHAPE))
    model.add(Cropping2D(((50, 20), (0, 0))))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))

    return model


def nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=INPUT_SHAPE))
    model.add(Cropping2D(((50, 20), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model


def comma_model():
    weight_init = 'he_normal'
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=INPUT_SHAPE))
    model.add(Cropping2D(((50, 20), (0, 0))))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', init=weight_init))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same', init=weight_init))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same', init=weight_init))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(ELU())
    model.add(Dense(512, init=weight_init))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1))

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

            yield shuffle(np.array(batch_images), np.array(batch_steerings))


def get_model(model_name):
    try:
        model = load_model(model_name + '.h5')
        print('Load from trained model')
    except:
        model = globals()[model_name]()
        model.compile(loss='mse', optimizer='adam')
        print('train new model')

    return model


def train(sources, model_name, epochs=EPOCHS):
    driving_logs = pd.concat([read_driving_log(source) for source in sources])

    model = get_model(model_name)
    print(model.summary())

    train_data, validation_data = train_test_split(driving_logs, test_size=0.2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
    save_weights = ModelCheckpoint(model_name + '.h5', monitor='val_loss', save_best_only=True)

    history = model.fit_generator(data_generator(train_data, BATCH_SIZE, augment=True),
                                  samples_per_epoch=len(train_data) * 6,
                                  nb_epoch=epochs,
                                  validation_data=data_generator(validation_data, BATCH_SIZE, augment=False),
                                  nb_val_samples=len(validation_data) * 3,
                                  callbacks=[save_weights, early_stopping])

    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    data = []
    data.append('track1')
    data.append('test2')
    # data.append('test2_r')

    train(data, 'nvidia_model', 5)
