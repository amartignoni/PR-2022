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
    # Main datastructure to store the preprocessed data
    preprocessed_data = pd.DataFrame(columns=["id", "transcription", "image"])

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
    width = 800
    height = 200

    for id, polygon in polygons.items():
        # The extraction of the images was inspired by this post in Stackoverflow
        # https://stackoverflow.com/questions/30901019/extracting-polygon-given-coordinates-from-an-image-using-opencv

        # Create the mask and apply it to the original image
        mask = np.zeros((orig_height, orig_width))
        cv.fillConvexPoly(mask, polygon, 1)
        mask = mask.astype(bool)
        out = np.zeros_like(binarized_img)
        out[mask] = binarized_img[mask]

        # Translate the present pixels in the center of the original image
        mean_x, mean_y = polygon.mean(axis=0)
        center_x, center_y = (out.shape[1] / 2, out.shape[0] / 2)
        offset_x, offset_y = (-mean_x + center_x, -mean_y + center_y)
        mx, my = np.meshgrid(np.arange(out.shape[1]), np.arange(out.shape[0]))
        ox = (mx - offset_x).astype(np.float32)
        oy = (my - offset_y).astype(np.float32)
        out_translate = cv.remap(out, ox, oy, cv.INTER_LINEAR)

        # Positions to crop
        left = center_x - width / 2
        right = center_x + width / 2
        top = center_y - height / 2
        bot = center_y + height / 2
        # Conversion to integer
        left, right, top, bot = np.floor([left, right, top, bot]).astype(np.int32)
        # Crop the image using slicing
        crop = out_translate[top:bot, left:right]

        # Add to preprocessed data: id, transcription and preprocessed image
        # Need to pass an index to concatenate below
        new_row = pd.DataFrame(
            {
                "id": id,
                "transcription": transcriptions[id],
                # Data needs to be 1D so wrap it in a list
                "image": [crop],
            },
            index=[0],
        )

        preprocessed_data = pd.concat([preprocessed_data, new_row], ignore_index=True)
        print(".", end="", flush=True)

    # Current solution to avoid not enough RAM -> one csv for each file
    # TODO: store preprocessed data in another structure? What is the best for the tasks that come after?
    preprocessed_data.to_csv(output_path / f"{svg_path.stem}.csv", index=False)
    print(f"{svg_path.stem} done!")

# Christophe: Preprocessing
# Dominik: Features
# Augustin: DTW/algorithms
