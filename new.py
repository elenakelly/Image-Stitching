### Import libraries ###
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio

### SIFT matching ###
feature_external_algo = 'sift'
feature_to_match = 'bf'

### Load images ###
train_image = cv2.imread('images/rightimage.png')
train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
train_image_gray = cv2.cvtColor(train_image, cv2.COLOR_RGB2GRAY)

test_image = cv2.imread('images/leftimage.png')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

fig, (ax2, ax1) = plt.subplots(1, 2, constrained_layout = False ,figsize=(20, 10))
ax1.imshow(test_image, cmap='gray')
ax1.set_title('Test Image')
ax2.imshow(train_image, cmap='gray')
ax2.set_title('Train Image')
plt.show()

### Harris corner detection ###
train_image_gray = np.float32(train_image_gray)
keypoints1 = cv2.cornerHarris(train_image_gray, 2, 3, 0.01)
keypoints1 = cv2.dilate(keypoints1, None, iterations=5)

test_image_gray = np.float32(test_image_gray)
keypoints2 = cv2.cornerHarris(test_image_gray, 2, 3, 0.01)
keypoints2 = cv2.dilate(keypoints2, None, iterations=5)

print("train_image: ",keypoints1,"test_image: " ,keypoints2)

#gets corners above a specified threshold
train_image[keypoints1 > 0.01 * keypoints1.max()] = [0, 0, 255]
test_image[keypoints2 > 0.01 * keypoints2.max()] = [0, 0, 255]

fig, (ax2, ax1) = plt.subplots(1, 2, constrained_layout = False ,figsize=(20, 10))
ax1.imshow(test_image, cmap='gray')
ax1.set_title('Test Image')
ax1.axis('off')
ax2.imshow(train_image, cmap='gray')
ax2.set_title('Train Image')
ax2.axis('off')
plt.show()

print("Number of corners in train image: ", np.sum(keypoints1 > 0.01 * keypoints1.max()))
print("Number of corners in test image: ", np.sum(keypoints2 > 0.01 * keypoints2.max()))

#keypoints2 = np.float32([kp.pt for kp in keypoints2])

#img = cv2.drawKeypoints(test_image_gray, keypoints1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite("images/train_image.png", img)

### SIFT descriptors ###
def select_descriptors(image, kps, method= None):
    kps = np.float32([kp.ptp for kp in kps])
    image8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    if method == 'sift' :
        sift = cv2.xfeatures2d.SIFT_create()
    (kp, des) = sift.detectAndCompute(image8bit, kps)
    print("SIFT descriptors: ", des)
    return (kp, des)

#kp_train, des_train = select_descriptors(train_image, method= feature_external_algo)

kp_test, des_test = select_descriptors(test_image, method= feature_external_algo, kps=keypoints2)


print("Number of keypoints in test image: ", len(kp_test))
#print("Number of keypoints in train image: ", len(kp_train))

'''#img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('sift_keypoints.jpg',img)
img = cv2.imread('home.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp)
cv2.imwrite('sift_keypoints.jpg',img)'''

for kp in kp_test:
    x, y = kp.pt
    size = kp.size
    orientation = kp.angle
    response = kp.response
    octave = kp.octave
    class_id = kp.class_id
print("x: ", x, "y: ", y, "size: ", size, "orientation: ", orientation, "response: ", response, "octave: ", octave, "class_id: ", class_id)

### Normalized Correlation ###

### Euclidian distance computation ###

### Compute the distance ###



### RANSAC ###

def ransac_match(kp_train, kp_test, matches, reprojThres):
    kp_train = np.float32([kp.pt for kp in kp_train])
    kp_test = np.float32([kp.pt for kp in kp_test])
    if len(matches) > 4:
        points_train = np.float32([kp_train[m.trainIdx].pt for m in matches])
        points_test = np.float32([kp_test[m.queryIdx].pt for m in matches])

        (H, status) = cv2.findHomography(points_train, points_test, cv2.RANSAC, reprojThres)
        
        return (matches, H, status)
    else:
         return None

M = ransac_match(kp_train, kp_test, matches, reprojThres=4.0) 
if M is not None:
    print('Error')

(matches, H, status) = ransac_match(kp_train, kp_test, matches, reprojThres=4.0)
print(H)

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