import cv2 as cv
import numpy as np
from pathlib import Path
from xml.dom import minidom
# from cairosvg import svg2png


# Conversion from SVG path to list of contour points
def path_to_contour(svg_path):
    splited = svg_path.split()
    splited = [float(elem) for elem in splited if elem not in ['M', 'L', 'Z']]
    # pairwise iteration
    points = np.array([[x, y] for x, y in zip(splited[0::2], splited[1::2])], dtype=np.int32)
    return points

root_path = Path.cwd().parents[0] / 'PatRec17_KWS_Data'
# TODO: iterate over all files in a directory an not use id anymore, this is just to test
id = 275
svg_path = root_path / 'ground-truth' / 'locations' / f"{id}.svg"
img_path = root_path / 'images' / f"{id}.jpg"

doc = minidom.parse(str(svg_path))
# Dict containing id (key) and list of contour points (value)
polygons = {path.getAttribute('id') : path_to_contour(path.getAttribute('d')) for path in doc.getElementsByTagName('path')}
doc.unlink()
# print(polygons)

# imread needs a string
text_img = cv.imread(str(img_path))
for _, poly in polygons.items():
    # List of points needs to be itself in a list
    cv.polylines(text_img, [poly], isClosed=True, color=(0,0,255), thickness=2)
cv.imshow("Display window", text_img)
# Press a key to close the window
cv.waitKey(0)
