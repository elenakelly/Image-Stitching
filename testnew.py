import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy 


img1 = cv2.imread('images/leftimage.png') #query image
img2= cv2.imread('images/rightimage.png') #train image
cv2.imshow('img1', img1)
#cv2.waitKey(0)
cv2.imshow('img2', img2)
#cv2.waitKey(0)

#detect Haris corners for img1
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray1 = np.float32(gray1)
dst1 = cv2.cornerHarris(gray1, 2, 3, 0.01)
dst1 = cv2.dilate(dst1, None)
a = dst1 > 0.01 * dst1.max()
x,y = np.where(a==True)

img1[a] = [0, 0, 255]
cv2.imshow('img1', img1)
#cv2.waitKey(0)

img2[a] = [0, 0, 255]
cv2.imshow('img1', img2)
#cv2.waitKey(0)


#convert to keypoint object
eye_corner_cordinates = []
for i in np.arange(len(x)):
    eye_corner_cordinates.append([x[i],y[i]])
x= np.float32(x)

kp1 = [cv2.KeyPoint(float(crd[0]),float(crd[1]),3) for crd in eye_corner_cordinates]

#compute SIFT descriptors for corner keypoiints
sift = cv2.SIFT_create()
kp1, des1 = sift.compute(img1, kp1)

#detect Haris corners for img2
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray2 = np.float32(gray2)
dst2 = cv2.cornerHarris(gray2, 2, 3, 0.01)
dst2 = cv2.dilate(dst2, None)
a = dst2 > 0.01 * dst2.max()
x,y = np.where(a==True)
eye_corner_cordinates = []
for i in np.arange(len(x)):
    eye_corner_cordinates.append([x[i],y[i]])
kp2 = [cv2.KeyPoint(float(crd[0]),float(crd[1]),3) for crd in eye_corner_cordinates]

#compute SIFT descriptors for corner keypoiints
sift = cv2.SIFT_create()
#des2 = [sift.compute(img2, kp)[1] for kp in kp1]
kp2, des2 = sift.compute(img2, kp2)

ds1 = []
ds2 = []
for i in np.arange(len(des1)):
    ds1.append(des1[i][0])
for i in np.arange(len(des2)):
    ds2.append(des2[i][0])
ds1 = np.array(ds1)
ds2 = np.array(ds2)

# Convert keypoints from numpy arrays to list of OpenCV KeyPoint objects
'''xy_pairs = list(kp1)
cv2_kpts = []
for i in range(len(xy_pairs)):
    cv2_kpts.append( cv2.KeyPoint(xy_pairs[i][0], xy_pairs[i][1], 1) )'''
    

# normalize the descriptors
for i in range(ds1.shape[0]):
        ds1[i,:] = ds1[i,:]/np.linalg.norm(ds1[i,:],ord=2)

for i in range(ds2.shape[0]):
        ds2[i,:] = ds2[i,:]/np.linalg.norm(ds2[i,:],ord=2)

# compute euclidean distances between descriptors
distance_matrix = scipy.spatial.distance_matrix(ds1, ds2)

# compute correlation between every img1 and img2 normalized descriptors
correlation_matrix = np.dot(ds1,ds2.T)




#dist_l2  = cv2.NORM_L2(ds1, ds2, cv2.NORM_L2)

#bf = cv2.BFMatcher()
#matches = bf.knnMatch(ds1, ds2, k=2)
#img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags=2)
#img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
#plt.imshow(img3)
#plt.show()







