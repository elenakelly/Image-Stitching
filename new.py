### Import libraries ###
from turtle import clear, width
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio

feature_external_algo = 'sift'
feature_to_match = 'bf'

### Load images ###
train_image = cv2.imread('images/rightimage.png')
train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
train_image_gray = cv2.cvtColor(train_image, cv2.COLOR_RGB2GRAY)

test_image = cv2.imread('images/leftimage.png')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout = False ,figsize=(20, 10))
ax1.imshow(test_image, cmap='gray')
ax1.set_title('Test Image')
ax2.imshow(train_image, cmap='gray')
ax2.set_title('Train Image')
plt.show()

### Harris corner detection ###
train_image_gray = np.float32(train_image_gray)
dst = cv2.cornerHarris(train_image_gray, 2, 3, 0.04)


### SIFT descriptors ###
def select_descriptors(image, method= None):
    image8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    if method == 'sift' :
        sift = cv2.SIFT_create()
    (kp, des)= sift.detectAndCompute(image8bit, None)
    return (kp, des)
kp_test, des_test = select_descriptors(test_image_gray, method= feature_external_algo)
kp_train, des_train = select_descriptors(train_image_gray, method= feature_external_algo)

### RANSAC ###

### Wrap images ###
width = test_image.shape[1] + train_image.shape[1]
print("width" ,width)
height = max(test_image.shape[0], train_image.shape[0])
result = cv2.warpPerspective(train_image, np.eye(3), (width, height))
result[0:test_image.shape[0], 0:test_image.shape[1]] = test_image

plt.figure(figsize=(20, 10))
plt.axis('off')
plt.imshow(result)
plt.show()