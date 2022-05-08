# Image Stitching Application

A panorama application that is stitching together pairs of images.<br />
To use just run the main.py file. <br />
Within the program you are allowed to upload the wanted images into the image file.<br />
Then, change the route of the imported images to the disared ones <br />
The program is going to output all the steps the image takes to complete the stitching.<br /> 

## Methodology

- Detecting keypoints in all of the images with Harris corner detection.
- Computes SIFT descriptors on resulted key-points
- Matching the descriptors between the two images with Normalized correlation and Euclidian distance.
- Using RANSAC algorithm to estimate Affine transformation.
- Applying warp transformation.


## Installation
The program is in Python <br />
In order to use the code you need to install Numpy, Matplotlib and OpenCv libraries
   ```sh
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.spatial
   ```

## Contributors
Elena Kane </br>
