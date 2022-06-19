import numpy as np
import scipy.stats as stats

ID = "id"
FEATURES = "features"
IMAGE = "image"
TRANSCRIPTION = "transcription"


def get_features(image):
    image_features = []
    for col_idx in range(image.shape[1] - 1):
        image_features.append(
            calculate_feature_vector(image[:, col_idx], image[:, col_idx + 1])
        )
    return normalize(np.array(image_features))


# def normalize(feature_vectors):
#     transposed = feature_vectors.T
#     for idx, column in enumerate(transposed):
#         transposed[idx] = stats.zscore(column)
#     return transposed

# def normalize(feature_vectors):
#     transposed = feature_vectors.T

#     transposed_normed = (transposed - transposed.min(0)) / transposed.ptp(0)

#     return transposed_normed

def normalize(feature_vectors):

    normalized = (feature_vectors - feature_vectors.min(0)) / feature_vectors.ptp(0)
    # print(normalized, normalized.shape)

    return normalized.T

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
    idx = np.flatnonzero(window)

    if idx.size == 0:
        return window.shape[0] - 1

    return idx[0]


def lower_contour(window):
    idx = np.flatnonzero(window)

    if idx.size == 0:
        return 0

    return idx[-1]


def number_of_black_white_transitions(window):
    return np.count_nonzero(window[:-1] > window[1:])


def fraction_of_black_pixels(window):
    return np.count_nonzero(window) / len(window)


def fraction_of_black_pixels_between_uc_and_lc(window):
    if np.count_nonzero(window) == 0:
        return 0
    return np.count_nonzero(
        window[upper_contour(window): lower_contour(window)+1]
    ) / len(window[upper_contour(window): lower_contour(window)+1])


def gradient_lc(curr_window, next_window):
    return lower_contour(next_window) - lower_contour(curr_window)


def gradient_uc(curr_window, next_window):
    return upper_contour(next_window) - upper_contour(curr_window)
