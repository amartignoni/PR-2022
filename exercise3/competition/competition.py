import cv2 as cv
import sys
import numpy as np
from pathlib import Path
import csv
import os
from xml.dom import minidom

from scipy.spatial.distance import euclidean

sys.path.append("../features")
sys.path.append("..")
from features import get_features
from sympy.combinatorics import Permutation
#from fastdtw import fastdtw
from tslearn.metrics import dtw
from string_utils import correct_string



# Creating paths
root_path = Path.cwd().parents[0] / "competition" / "data"
svg_paths = root_path / "ground-truth" / "locations"
img_root = root_path / "images"
output_path_preprocessing = Path.cwd() / "output" / "preprocessing"
output_path_preprocessing.mkdir(parents=True, exist_ok=True)
output_path_distance = Path.cwd() / "output" / "distance"
output_path_distance.mkdir(parents=True, exist_ok=True)
valid_path = Path.cwd().parents[0] / "competition" / "output" / "preprocessing" / "valid"
valid_savepath = Path.cwd().parents[0] / "competition" / "output" / "distance" / "valid"
out_path = Path.cwd().parents[0] / "competition" / "output" / "distance" / "distances.csv"
transcription_path = root_path / "task" / "keywords.txt"
train_savepath = Path("../distance/output/train")


def path_to_contour(svg_path):
    splited = svg_path.split()
    splited = [float(elem) for elem in splited if elem not in ["M", "L", "Z"]]
    # Pairwise iteration
    points = np.array(
        [[x, y] for x, y in zip(splited[0::2], splited[1::2])], dtype=np.int32
    )
    return points


def bounding_box(points):

    x_coords, y_coords = zip(*points)

    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)


def load_files_and_compute_features(load_path, save_path):

    ids = []  # store arrays to stack together into one
    images = []

    for image in load_path.iterdir():

        filename = os.path.basename(image)

        img_id = os.path.splitext(filename)[0]

        ids.append(img_id)

        image_array = np.genfromtxt(image, delimiter=",", dtype="float64")

        feature_array = get_features(image_array)  # modify images into feature vectors

        images.append(feature_array)

        np.savetxt(
            Path(save_path / f"{img_id}.csv").as_posix(), feature_array, delimiter=",", fmt="%1f"
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

        feature_array = np.genfromtxt(image, delimiter=",", dtype="float64")

        images.append(feature_array)

    # sort ids and features lists the same way

    ids_sorting = [(ids[i], i) for i in range(len(ids))]
    ids_sorting.sort()

    split = [[i for i, j in ids_sorting], [j for i, j in ids_sorting]]
    ids = [x for x in split[0]]

    perm = Permutation([x for x in split[1]])

    images = perm(images)

    return ids, images


def generalizedDTW(test, training):

    #distance, _ = fastdtw(test, training, dist=euclidean)
    distance = dtw(test.T, training.T, global_constraint="sakoe_chiba")
    return distance


# PART 1: Images preprocessing
for svg_path in svg_paths.iterdir():

    # Parsing SVG
    # parse needs a string
    doc = minidom.parse(str(svg_path))
    # Dict containing id (e.g. 278-03-05) as key and the list of contour points (polygons) as value
    polygons = {
        path.getAttribute("id"): path_to_contour(path.getAttribute("d"))
        for path in doc.getElementsByTagName("path")
    }
    doc.unlink()

    # imread needs a string
    img_path = img_root / f"{svg_path.stem}.jpg"
    orig_img = cv.imread(str(img_path), 0)
    _, binarized_img = cv.threshold(orig_img, 170, 255, cv.THRESH_BINARY_INV)

    orig_width = binarized_img.shape[1]
    orig_height = binarized_img.shape[0]
    # Size of output preprocessed image
    width = 100
    height = 100

    for id_, polygon in polygons.items():
        left, right, top, bot = bounding_box(polygon)

        out = binarized_img[top:bot, left:right]

        mask = np.ones(out.shape)

        cv.fillPoly(mask, [polygon], 0)

        out = np.where(out > 0, 1, out)

        out = out * mask

        out = cv.resize(out, (200, 200), interpolation=cv.INTER_NEAREST)

        out = out.astype(int)

        print(".", end="", flush=True)

        np.savetxt(
            Path(output_path_preprocessing / "valid" / f"{id_}.csv").as_posix(),
            out,
            delimiter=",",
            fmt="%1i",
        )

    print(f"{svg_path.stem} done!")


# PART 2: distance calculation
train_ids, train_features = load_precomputed_features(train_savepath)

valid_ids, valid_features = load_files_and_compute_features(
        valid_path, valid_savepath
    )

dist_mat = np.array(
        [[generalizedDTW(i, j) for j in train_features] for i in valid_features]
    )

np.savetxt("./output/mat.csv", dist_mat, delimiter=",", fmt="%1f")


## generate csv

def get_transcriptions():
    transcriptions = {}
    file = open(transcription_path, "r")
    for line in file:
        transcription, img_id = line.split()

        if img_id[0:3] in ["300", "301", "302", "303", "304"]:
            transcriptions[img_id] = correct_string(transcription)

    sorted_transcriptions = dict(
        sorted(transcriptions.items(), key=lambda item: item[0])
    )

    return sorted_transcriptions

out = []

transcriptions = get_transcriptions()

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

