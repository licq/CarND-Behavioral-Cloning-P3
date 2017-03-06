import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.model_selection import train_test_split

import utils

MODEL_FILE = 'model.h5'
EPOCHS = 5
BATCH_SIZE = 64


def get_model(model_func):
    try:
        model = load_model(MODEL_FILE)
        print('Load from trained model')
    except:
        model = model_func()
        print('train new model')

    return model


def main():
    # track1 = utils.read_driving_log('track1')
    # track2 = utils.read_driving_log('track2', has_header=False)
    # driving_log = pd.concat([track1, track2])

    track1 = utils.read_driving_log('track1')
    # test2 = utils.read_driving_log('test2', has_header=False)
    # test2_r = utils.read_driving_log('test2_r', has_header=False)
    # driving_log = pd.concat([track1, test2])
    model = get_model(utils.openai_model)
    driving_log = track1

    train, validation = train_test_split(driving_log, test_size=0.2)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
    save_weights = ModelCheckpoint(MODEL_FILE, monitor='val_loss', save_best_only=True)

    # model.fit_generator(utils.random_data_generator(train, BATCH_SIZE, augment=True),
    #                     samples_per_epoch=BATCH_SIZE * 400,
    #                     nb_epoch=EPOCHS,
    #                     validation_data=utils.random_data_generator(validation, BATCH_SIZE, augment=False),
    #                     nb_val_samples=BATCH_SIZE * 10)
    model.fit_generator(utils.data_generator(train, BATCH_SIZE, augment=True),
                        samples_per_epoch=len(train) * 6,
                        nb_epoch=EPOCHS,
                        validation_data=utils.data_generator(validation, BATCH_SIZE, augment=False),
                        nb_val_samples=len(validation) * 3,
                        callbacks=[save_weights, early_stopping])


if __name__ == '__main__':
    main()
