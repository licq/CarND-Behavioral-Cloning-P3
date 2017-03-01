import pandas as pd
from sklearn.model_selection import train_test_split

import utils

DRIVING_LOG_CSV = 'data/driving_log.csv'
EPOCHS = 5
BATCH_SIZE = 64


def main():
    driving_log = pd.read_csv(DRIVING_LOG_CSV)
    # driving_log = utils.sample(driving_log, 0.1, 1000)

    model = utils.nvidia_model((160, 320, 3))

    train, validation = train_test_split(driving_log, test_size=0.2)

    model.fit_generator(utils.data_generator(train, BATCH_SIZE, augment=True),
                        samples_per_epoch=BATCH_SIZE * 200,
                        nb_epoch=EPOCHS,
                        validation_data=utils.data_generator(validation, BATCH_SIZE, augment=False),
                        nb_val_samples=BATCH_SIZE*10)
    model.save('model.h5')


if __name__ == '__main__':
    main()
