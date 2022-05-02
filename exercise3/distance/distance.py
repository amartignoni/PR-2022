import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import torch as th
import tslearn as tl

## plan

## compute features for all images (training and testing data)

# Load CSV

## function to compute dtw between test image and array of training images

# test : n.features x width ; training : n.samples x n.features x width
def generalizedDTW(test, training):
    return tl.metrics.dtw(test, training, 'sakoe_chiba')

## sort images and return best guess (or best guesses)

# WordMatcher class implementation adapted from : https://gist.github.com/JosueCom/7e89afc7f30761022d7747a501260fe3

class WordMatcher():

    def __init__(self, X = None, Y = None, Z = None, k = 1):
        self.train(X, Y, Z)
        self.k = k
    
    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y
        self.train_IDs = Z

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = generalizedDTW(x, self.train_pts)

        # sort and return k best

        dist_sorted, indices = th.sort(dist)
        labels_sorted = self.train_label[indices]
        IDs_sorted = self.train_IDs[indices]

        return th.cat((labels_sorted[:k], dist_sorted[:k], IDs_sorted[:K]), 0)


if __name__=='main':


