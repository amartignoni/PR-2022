import numpy as np
from pathlib import Path
import csv
import os
import sys

sys.path.append("../features")
from features import get_features
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sympy.combinatorics import Permutation
import string

train_path = Path.cwd().parents[0] / "preprocessing" / "output" / "train"
valid_path = Path.cwd().parents[0] / "preprocessing" / "output" / "valid"
transcription_path = (
    Path.cwd().parents[0] / "data" / "ground-truth" / "transcription.txt"
)
train_savepath = Path.cwd().parents[0] / "distance" / "output" / "train"
valid_savepath = Path.cwd().parents[0] / "distance" / "output" / "valid"
out_path = Path.cwd().parents[0] / "distance" / "output" / "distances.csv"

## compute features for all images (training and testing data)


def load_files_and_compute_features(load_path, save_path):

    ids = []  # store arrays to stack together into one
    images = []

    for image in load_path.iterdir():

        filename = os.path.basename(image)

        img_id = os.path.splitext(filename)[0]

        ids.append(img_id)

        image_array = np.genfromtxt(image, delimiter=",", dtype="uint8")

        feature_array = get_features(image_array)  # modify images into feature vectors

        images.append(feature_array)

        np.savetxt(
            Path(save_path / f"{img_id}.csv"), feature_array, delimiter=",", fmt="%1i"
        )

    # sort ids and features lists the same way

    ids_sorting = [(ids[i], i) for i in range(len(ids))]
    ids_sorting.sort()

    split = [[i for i, j in ids_sorting], [j for i, j in ids_sorting]]
    ids = [x for x in split[0]]

    perm = Permutation([x for x in split[1]])

    images = perm(images)

    return ids, images


def load_precomputed_features(load_path):

    ids = []
    images = []

    for image in load_path.iterdir():

        filename = os.path.basename(image)

        img_id = os.path.splitext(filename)[0]

        ids.append(img_id)

        feature_array = np.genfromtxt(image, delimiter=",", dtype="uint8")

        images.append(feature_array)

    # sort ids and features lists the same way

    ids_sorting = [(ids[i], i) for i in range(len(ids))]
    ids_sorting.sort()

    split = [[i for i, j in ids_sorting], [j for i, j in ids_sorting]]
    ids = [x for x in split[0]]

    perm = Permutation([x for x in split[1]])

    images = perm(images)

    return ids, images


## function to compute dtw between two arrays of features


def generalizedDTW(test, training):

    distance, _ = fastdtw(test, training, dist=euclidean)

    return distance


def correct_string(str):

    str_ = str.split("-")

    if len(str_) == 1:

        if str_[0] == "s_GW":

            return "GW"

        elif str_[0] == "s_mi":

            return "-"

    str_corrected = [
        char[2:]
        if char
        in [
            "s_0",
            "s_1",
            "s_2",
            "s_3",
            "s_4",
            "s_5",
            "s_6",
            "s_7",
            "s_8",
            "s_9",
            "s_0th",
            "s_1st",
            "s_2nd",
            "s_3rd",
            "s_4th",
            "s_5th",
            "s_6th",
            "s_7th",
            "s_8th",
            "s_9th",
            "s_s",
        ]
        else char
        for char in str_
    ]

    # correct signature

    alphanumeric = list(string.ascii_letters) + list(string.digits)

    str_without_specials = filter(
        lambda substr: all([char in alphanumeric for char in substr]), str_corrected
    )

    final_str = "".join(str_without_specials)

    return final_str


def get_valid_transcriptions():

    transcriptions = {}

    file = open(transcription_path, "r")

    for line in file:

        img_id, transcription = line.split()

        if img_id[0:3] in ["300", "301", "302", "303", "304"]:

            transcription = correct_string(transcription)

            if transcription == "":

                print(img_id)

            transcriptions[img_id] = transcription

    sorted_transcriptions = dict(
        sorted(transcriptions.items(), key=lambda item: item[0])
    )

    return sorted_transcriptions


## sort images and return best guess (or best guesses)

# compute distances

if (
    sys.argv[1] == "1"
):  # pass 1 as first argument to compute features, anything else to load precomputed

    train_ids, train_features = load_files_and_compute_features(
        train_path, train_savepath
    )

    valid_ids, valid_features = load_files_and_compute_features(
        valid_path, valid_savepath
    )

else:

    train_ids, train_features = load_precomputed_features(train_savepath)

    valid_ids, valid_features = load_precomputed_features(valid_savepath)

if (
    sys.argv[2] == "1"
):  # pass 1 as second argument to compute distance matrix, anything else to load precomputed

    dist_mat = np.array(
        [[generalizedDTW(i, j) for j in train_features] for i in valid_features]
    )

    np.savetxt("./output/mat.csv", dist_mat, delimiter=",", fmt="%1f")

else:

    dist_mat = np.genfromtxt("./output/mat.csv", delimiter=",", dtype="float64")


## generate csv

out = []

transcriptions = get_valid_transcriptions()

for i in range(len(valid_features)):

    keyword = transcriptions[valid_ids[i]]

    temp_dict = {k: v for (k, v) in zip(train_ids, dist_mat[i, :])}

    tuple_list = sorted(temp_dict.items(), key=lambda x: x[1])

    flattened = [item for kv_tuple in tuple_list for item in kv_tuple]

    flattened.insert(0, keyword)

    out.append(flattened)

with open(out_path.as_posix(), "w") as csvfile:

    writer = csv.writer(csvfile, delimiter=",")

    writer.writerows(out)
