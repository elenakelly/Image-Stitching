# Image Stitching
A panorama application that is stitching together pairs of images

- Detecting keypoints in all of the images with Harris corner detection.
- Computes SIFT descriptors on resulted key-points
- Normalized correlation and Euclidian distance
- Matching the descriptors between the two images.
- Using RANSAC algorithm to estimate Affine transformation.
- Applying warp transformation.


