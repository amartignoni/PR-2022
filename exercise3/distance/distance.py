import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import torch as th
import tslearn as tl
import csv
import sys
import os
sys.path.append('../features')
from features import get_features
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from sympy.combinatorics import Permutation

train_path = Path.cwd().parents[0] / "preprocessing" / "output" / "train"
valid_path = Path.cwd().parents[0] / "preprocessing" / "output" / "valid"
transcription_path = Path.cwd().parents[0] / "data" / "ground-truth" / "transcription.txt"
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

        image_array = np.genfromtxt(image, delimiter = ',', dtype='uint8')

        feature_array = get_features(image_array)  # modify images into feature vectors

        images.append(feature_array)

        np.savetxt(Path(save_path / f"{img_id}.csv"), feature_array, delimiter=",", fmt='%1i')

    # sort ids and features lists the same way

    ids_sorting = [(ids[i],i) for i in range(len(ids))]
    ids_sorting.sort()

    split = [[i for i,j in ids_sorting],[j for i,j in ids_sorting]]
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

        feature_array = np.genfromtxt(image, delimiter = ',', dtype='uint8')

        images.append(feature_array)

    # sort ids and features lists the same way

    ids_sorting = [(ids[i],i) for i in range(len(ids))]
    ids_sorting.sort()

    split = [[i for i,j in ids_sorting],[j for i,j in ids_sorting]]
    ids = [x for x in split[0]]

    perm = Permutation([x for x in split[1]])

    images = perm(images)

    return ids, images


## function to compute dtw between two arrays of features

def generalizedDTW(test, training):
    #return tl.metrics.dtw(test, training, "sakoe_chiba")
    distance, _ = fastdtw(test, training, dist=euclidean)

    print(distance)
    
    return distance

def get_valid_transcriptions():

    transcriptions = {}

    file = open(transcription_path, "r")
    
    for line in file:
        
        img_id, transcription = line.split()

        if img_id[0:3] in ["300","301","302","303","304"]:

            transcriptions[id] = transcription

    sorted_transcriptions = dict(sorted(transcriptions.items(), key=lambda item: item[0]))

    return sorted_transcriptions

## sort images and return best guess (or best guesses)

# compute distances

if sys.argv[1] == "1":
    
    train_ids, train_features = load_files_and_compute_features(train_path, train_savepath)
    
    valid_ids, valid_features = load_files_and_compute_features(valid_path, valid_savepath)
    
else:
    
    train_ids, train_features = load_precomputed_features(train_savepath)
    
    valid_ids, valid_features = load_precomputed_features(valid_savepath)

if sys.argv[2] == "1":    
    
    dist_mat = np.array(
        [
            [generalizedDTW(i, j) for j in train_features]
            for i in valid_features
        ]
    )
    
    np.savetxt("./mat.csv", dist_mat, delimiter=',')

else:

    dist_mat = np.genfromtxt("./mat.csv", delimiter = ',', dtype='uint8')
    
    
## generate csv

out = []

transcriptions = get_valid_transcriptions()

print(transcriptions)

for i in range(len(valid_features)):

    keyword = transcriptions[valid_ids[i]]

    temp_dict = {}

    for k, v in zip(train, dist_mat[i]):

        temp_dict[k] = v

    temp_dict_sorted = sorted(temp_dict.items(), key = lambda x : x[1])

    tuple_list = temp_dict_sorted.items()

    flattened = [y for x in tuple_list for y in x]

    transcriptions.insert(0, keyword)

    out.append(transcriptions)

with open(out_path.as_posix(), 'wb') as csvfile:

    filewriter = csv.writer(csvfile, delimiter=',')

    for lst in out: 

        filewriter.writerow(lst)









