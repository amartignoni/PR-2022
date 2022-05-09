import numpy as np
import pandas as pd

ID = "id"
FEATURES = "features"
IMAGE = "image"
TRANSCRIPTION = "transcription"


def get_features(preprocessed_data):
    features = []
    for image_object in preprocessed_data:
        image_id = image_object[ID]
        image = image_object[IMAGE]
        image_transcription = image_object[TRANSCRIPTION]
        image_features = []
        for column in range(image.shape[1] - 1):
            image_features.append(
                calculate_feature_vector(image[:, column], image[:, column + 1])
            )
        feature_object = pd.DataFrame(
            {
                ID: image_id,
                TRANSCRIPTION: image_transcription,
                FEATURES: normalize(np.array(image_features)),
            }
        )
        features.append(feature_object)
    return features


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
    return window.tolist().index(1)


def lower_contour(window):
    idx = window[::-1].tolist().index(1)
    return window.shape[1] - idx - 1


def number_of_black_white_transitions(window):
    return np.count_nonzero(window[:-1] < window[1:])


def fraction_of_black_pixels(window):
    return np.count_nonzero(window == 0) / len(window)


def fraction_of_black_pixels_between_uc_and_lc(window):
    return np.count_nonzero(
        window[upper_contour(window) : lower_contour(window)] == 0
    ) / len(window)


def gradient_lc(window, next_window):
    return lower_contour(window) - lower_contour(next_window)


def gradient_uc(window, next_window):
    return upper_contour(window) - upper_contour(next_window)
