import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import cv2

data_path = 'data/'


def sample(df, limit, nb_small):
    small = df[np.abs(df['steering']) < limit]
    big = df[np.abs(df['steering']) >= limit]
    small_choice = np.random.choice(small.shape[0], nb_small)

    return shuffle(pd.concat([big, small.iloc[small_choice]]))


def read_images(line):
    left = cv2.imread(data_path + line['left'].strip())
    center = cv2.imread(data_path + line['center'].strip())
    right = cv2.imread(data_path + line['right'].strip())

    return left, center, right


def flip_image(image):
    return cv2.flip(image, 1)


def convert_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
