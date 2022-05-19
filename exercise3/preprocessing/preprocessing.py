import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from xml.dom import minidom


# Conversion from SVG path to list of contour points
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


# Creating paths
root_path = Path.cwd().parents[0] / "data"
svg_paths = root_path / "ground-truth" / "locations"
img_root = root_path / "images"
transcription_path = root_path / "ground-truth" / "transcription.txt"
output_path = Path.cwd() / "output"
output_path.mkdir(parents=True, exist_ok=True)


# PART 1: Transcriptions preprocessing
transcriptions = {}
file = open(transcription_path, "r")
for line in file:
    id, transcription = line.split()
    transcriptions[id] = transcription


# PART 2: Images preprocessing
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

        out = out * mask

        out = cv.resize(out, (200, 200), interpolation=cv.INTER_LINEAR)

        out = out.astype(int)

        print(".", end="", flush=True)

        if id_[0:3] in ["300", "301", "302", "303", "304"]:

            np.savetxt(
                Path(output_path / "valid" / f"{id_}.csv").as_posix(),
                out,
                delimiter=",",
                fmt="%1i",
            )

        else:

            np.savetxt(
                Path(output_path / "train" / f"{id_}.csv").as_posix(),
                out,
                delimiter=",",
                fmt="%1i",
            )

    print(f"{svg_path.stem} done!")

# Christophe: Preprocessing
# Dominik: Features
# Augustin: DTW/algorithms
