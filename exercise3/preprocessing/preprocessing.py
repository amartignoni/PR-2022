import cv2 as cv
import numpy as np
from pathlib import Path
from xml.dom import minidom
import math


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
# IDEA : use the parser to grab IDs from the SVG ?
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
text_img = cv.imread(str(img_path), 0) # load binary image
_, final_img = cv.threshold(text_img, 180, 255, cv.THRESH_BINARY)
# TODO: convert img to boolean array
for polygon in polygons.values():
    # List of points needs to be itself in a list
    # TODO: change the method to cv.fillConvexPoly()
    # https://stackoverflow.com/questions/30901019/extracting-polygon-given-coordinates-from-an-image-using-opencv
    cv.polylines(final_img, [polygon], isClosed=True, color=(0,0,255), thickness=4)
    # TODO: extract the image of the word -> masked array -> conversion to pillow
    # https://stackoverflow.com/questions/22588074/polygon-crop-clip-using-python-pil
    # TODO: resize the image, think about the size 50x200? (cv.resize()) -> convert back to array
    # TODO: save this array

# TODO: Load the transcription.txt and assign transcription to the array

#print(final_img.shape, final_img, np.average(text_img[1100:1160, 765:800]))
cv.imshow("Display", final_img)
# Press a key to close the window
cv.waitKey(0)
cv.destroyAllWindows()


# Christophe: Preprocessing
# Dominik: Features
# Augustin: DTW/algorithms
