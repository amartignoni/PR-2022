import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import torch as th
import tslearn as tl
import csv

train_path = Path.cwd().parents[0] / "preprocessing" / "output" / "train"
valid_path = Path.cwd().parents[0] / "preprocessing" / "output" / "valid"

train_savepath = Path.cwd().parents[0] / "distances" / "output" / "train.csv"
valid_savepath = Path.cwd().parents[0] / "distances" / "output" / "valid.csv"

## plan

## compute features for all images (training and testing data)


def load_files_and_compute_features(load_path, save_path):

    t = tuple()  # store arrays to stack together into one

    for image in load_path.iterdir():

        with open(image, "r") as f:  # read each csv into dictionary

            dico = [
                {k: v for k, v in row.items()}
                for row in csv.DictReader(f, skipinitialspace=True)
            ]

        compute_features = get_features(dico)  # modify images into feature vectors

        dict_to_array = np.array(  # transform each dict to a numpy array
            [[val for val in elem.values()] for elem in compute_features]
        )

        t.append(dict_to_array)

    res = np.cat(t)  # final array

    df = pd.DataFrame(res)

    df.to_csv(save_path)

    return res


def load_precomputed_features(load_path):

    res = np.loadtxt(load_path)

    return res


## function to compute dtw between test image and array of training images

# test : n.features x width ; training : n.samples x n.features x width


def generalizedDTW(test, training):
    return tl.metrics.dtw(test, training, "sakoe_chiba")


## sort images and return best guess (or best guesses)

# WordMatcher class implementation adapted from : https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3


class WordMatcher:
    def __init__(self, X=None, Y=None, Z=None, k=1):
        self.train(X, Y, Z)
        self.k = k

    def train(self, X, Y, Z):
        self.train_imgs = X
        self.train_labels = Y
        self.train_IDs = Z

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_imgs) == type(None) or type(self.train_labels) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(
                f"{name} wasn't trained. Need to execute {name}.train() first"
            )

        dist = generalizedDTW(x, self.train_imgs)

        # sort and return k best

        dist_sorted, indices = th.sort(dist)
        labels_sorted = self.train_label[indices]
        IDs_sorted = self.train_IDs[indices]

        return th.cat((labels_sorted[:k], dist_sorted[:k], IDs_sorted[:k]), 0)


if __name__ == "main":

    if sys.argv[1] == True:

        train = load_files_and_compute_features(train_path / "train.csv")

        valid = load_files_and_compute_features(valid_path / "valid.csv")

    else:

        train = load_precomputed_features(train_savepath)

        valid = valid_savepath(valid_savepath)

    WM = WordMatcher(train[:, 2], train[:, 1], train[:, 0], train.shape[0])

    # compute distance matrix

    # dist_mat = np.zeros((train.shape[0], valid.shape[0]))

    dist_mat = np.array(
        [
            [generalizedDTW(train[i, 0], valid[i, 0]) for i in range(train.shape[0])]
            for j in range(valid.shape[0])
        ]
    )

    # for i in train:
    #     for j in valid:
    #         dist_mat[i,j] = generalizedDTW(train[i, 0], valid[i, 0])


    ## generate csv
