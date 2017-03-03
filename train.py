import pandas as pd
from keras.models import model_from_json
from sklearn.model_selection import train_test_split

import utils

MODEL_WEIGHTS = 'model.h5'
MODEL_JSON = 'model.json'
EPOCHS = 10
BATCH_SIZE = 64


def get_model():
    try:
        with open(MODEL_JSON, 'r') as f:
            model = model_from_json(f.read())
        model.compile('adam', 'mse')
        model.load_weights(MODEL_WEIGHTS)
        print('Load from trained model')
    except:
        model = utils.nvidia_model((160, 320, 3), with_cropping=True)
        print('train new model')

    return model


def main():
    track1 = utils.read_driving_log('data')
    track2 = utils.read_driving_log('track2', has_header=False)
    driving_log = pd.concat([track1, track2])
    # driving_log = utils.sample(driving_log, 0.1, 1000)

    model = get_model()

    train, validation = train_test_split(driving_log, test_size=0.2)

    # model.fit_generator(utils.random_data_generator(train, BATCH_SIZE, augment=True),
    #                     samples_per_epoch=BATCH_SIZE * 400,
    #                     nb_epoch=EPOCHS,
    #                     validation_data=utils.random_data_generator(validation, BATCH_SIZE, augment=False),
    #                     nb_val_samples=BATCH_SIZE * 10)
    model.fit_generator(utils.data_generator(train, BATCH_SIZE, augment=True),
                        samples_per_epoch=len(train) * 6,
                        nb_epoch=EPOCHS,
                        validation_data=utils.data_generator(validation, BATCH_SIZE, augment=False),
                        nb_val_samples=len(validation) * 3)

    model.save_weights(MODEL_WEIGHTS)

    with open(MODEL_JSON, 'w') as f:
        f.write(model.to_json())


if __name__ == '__main__':
    main()
