import cv2 as cv
import numpy as np
from pathlib import Path
from xml.dom import minidom


# Conversion from SVG path to list of contour points
def path_to_contour(svg_path):
    splited = svg_path.split()
    splited = [float(elem) for elem in splited if elem not in ['M', 'L', 'Z']]
    # Pairwise iteration
    points = np.array([[x, y] for x, y in zip(splited[0::2], splited[1::2])], dtype=np.int32)
    return points

# Creating paths
root_path = Path.cwd().parents[0] / 'data'
# TODO: iterate over all files in a directory and not use id anymore, this is just to test
id = 278
svg_path = root_path / 'ground-truth' / 'locations' / f"{id}.svg"
img_path = root_path / 'images' / f"{id}.jpg"

# Parsing SVG
# parse needs a string
doc = minidom.parse(str(svg_path))
# Dict containing id (e.g. 278-03-05) as key and the list of contour points (polygons) as value
polygons = {path.getAttribute('id') : path_to_contour(path.getAttribute('d')) for path in doc.getElementsByTagName('path')}
doc.unlink()

# imread needs a string
text_img = cv.imread(str(img_path))
for polygon in polygons.values():
    # List of points needs to be itself in a list
    cv.polylines(text_img, [polygon], isClosed=True, color=(0,0,255), thickness=2)
cv.imshow("Display", text_img)
# Press a key to close the window
cv.waitKey(0)
