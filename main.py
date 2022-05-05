import numpy as np 
import cv2
import glob
import imutils

image_paths = glob.glob('images/*.png')
images = []

#storing each image path
for i in image_paths:
    img = cv2.imread(i)
    images.append(img)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

#stiching the images
imageStitcher = cv2.Stitcher_create()
(error, stitched_img) = imageStitcher.stitch(images)

if not error:
    cv2.imwrite("stitchedOutput.png", stitched_img)
    cv2.imshow("stitchedImage", stitched_img)
    cv2.waitKey(0)

    # post processing the stitched image

    #create a black border by 10 pixels
    stitched_img = cv2.copyMakeBorder(stitched_img, 10,10,10,10, cv2.BORDER_CONSTANT, (0,0,0))
    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY)[1]

    cv2.imwrite("thresholdImage.png", stitched_img)
    cv2.imshow("thresholdImage", threshold)
    cv2.waitKey(0)

    contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    max_contour_area = max(contours, key=cv2.contourArea)

    mask = np.zeros (threshold.shape, dtype='uint8')
    x,y,w,h = cv2.boundingRect(max_contour_area)
    cv2.rectangle(mask, (x,y), (x+w, y+h), 255,-1)

    min_rectangle = mask.copy()
    #subtract the min rectangle from the image
    sub = mask.copy()
    while cv2.countNonZero(sub)> 0:
        min_rectangle = cv2.erode(min_rectangle, None)
        sub = cv2.subtract(min_rectangle, threshold)
    
    contours =cv2.findContours(min_rectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours= imutils.grab_contours(contours)
    max_contour_area = max(contours, key=cv2.contourArea)

    cv2.imwrite("minRectangleImage.png", min_rectangle)
    cv2.imshow("minRectangleImage", min_rectangle)
    cv2.waitKey(0)

    x,y,w,h = cv2.boundingRect(max_contour_area)
    stitched_img = stitched_img[y:y+h, x:x + w]

    cv2.imwrite("stitchedOutputNew.png", stitched_img)
    cv2.imshow("stitchedOutputNew", stitched_img)
    cv2.waitKey(0)




