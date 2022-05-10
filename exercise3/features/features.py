import numpy as np
import pandas as pd

ID = "id"
FEATURES = "features"
IMAGE = "image"
TRANSCRIPTION = "transcription"


def get_features(image):
    image_features = []
    print(image)
    for column in range(image.shape[1] - 1):
        image_features.append(
            calculate_feature_vector(image[:, column], image[:, column + 1])
        )
    return normalize(np.array(image_features))


def normalize(feature_vectors):
    transposed = feature_vectors.T
    for column in transposed:
        transposed[column] = (
            transposed[column] - transposed[column].mean()
        ) / transposed[column].std()
    return transposed.T


def calculate_feature_vector(window, next_window):
    return [
        upper_contour(window),
        lower_contour(window),
        number_of_black_white_transitions(window),
        fraction_of_black_pixels(window),
        fraction_of_black_pixels_between_uc_and_lc(window),
        gradient_lc(window, next_window),
        gradient_uc(window, next_window),
    ]


def upper_contour(window):
    
    idx = np.nonzero(window)

    print(idx)

    if idx == ():
        print(image.shape[1] - 1)

        return image.shape[1] - 1 

    return idx[0]


def lower_contour(window):

    idx = np.nonzero(window)

    print(idx)

    if idx == ():
        return 0 

    return idx[-1]



def number_of_black_white_transitions(window):
    return np.count_nonzero(window[:-1] < window[1:])


def fraction_of_black_pixels(window):
    return np.count_nonzero(window == 0) / len(window)


def fraction_of_black_pixels_between_uc_and_lc(window):
    return np.count_nonzero(
        window[upper_contour(window) : lower_contour(window)] 
    ) / len(window)


def gradient_lc(window, next_window):
    return lower_contour(window) - lower_contour(next_window)


def gradient_uc(window, next_window):
    return upper_contour(window) - upper_contour(next_window)
