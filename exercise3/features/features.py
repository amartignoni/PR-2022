import numpy as np


def calculate_feature_vector(window):
    return [upper_contour(window),
            lower_contour(window),
            number_of_black_white_transitions(window),
            fraction_of_black_pixels(window),
            fraction_of_black_pixels_between_uc_and_lc(window),
            gradient_lc_uc(window)]


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
    return np.count_nonzero(window[upper_contour(window):lower_contour(window)] == 0) / len(window)


# TODO implement
def gradient_lc_uc(window):
    return window
