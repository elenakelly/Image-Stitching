
from matplotlib import image
import numpy as np 
import cv2
import glob
import imutils

image_paths = glob.glob('images/*.jpeg')
images = []

#storing each image path
for i in image_paths:
    img = cv2.imread(i)
    images.append(img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

#stiching the images
imageStitcher = cv2.Stitcher_create()
error, stitched_img = imageStitcher.stitch(images)

if not error:
    cv2.imwrite("stitchedOutput.png", stitched_img)
    cv2.imshow("stitched image", stitched_img)
    cv2.waitKey(0)


