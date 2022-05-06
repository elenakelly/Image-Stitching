import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy.spatial

### Load the images ###
img1 = cv2.imread('images/leftimage.png') #query image
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2= cv2.imread('images/rightimage.png') #train image
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout = False ,figsize=(10, 5))
ax1.imshow(img1_rgb, cmap='gray')
ax1.set_title('(a)')
ax1.axis('off')
ax2.imshow(img2_rgb, cmap='gray')
ax2.set_title('(b)')
ax2.axis('off')
plt.show()

### create sift object ###
sift = cv2.xfeatures2d.SIFT_create()
method = 'sift'

### Harris corner detection ###
img1_harris = cv2.cornerHarris(img1_gray, 2, 3, 0.04)
img1_harris = cv2.dilate(img1_harris, None)
ret, img1_harris = cv2.threshold(img1_harris, 0.05 * img1_harris.max(), 255, 0)
img1_harris = np.uint8(img1_harris)
_,_,_, centroids = cv2.connectedComponentsWithStats(img1_harris)
kp1 = centroids.copy().astype(np.uint16)

img2_harris = cv2.cornerHarris(img2_gray, 2, 3, 0.04)
img2_harris = cv2.dilate(img2_harris, None)
ret, img2_harris = cv2.threshold(img2_harris, 0.05 * img2_harris.max(), 255, 0)
img2_harris = np.uint8(img2_harris)
_,_,_, centroids = cv2.connectedComponentsWithStats(img2_harris)
kp2 = centroids.copy().astype(np.uint16)


### Visualize the results ###
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout = False ,figsize=(10, 5))
ax1.imshow(img1_rgb, cmap='gray')
ax1.plot(kp1[:,0], kp1[:,1], 'r*')
ax1.set_title('(a)')
ax1.axis('off')
ax2.imshow(img2_rgb, cmap='gray')
ax2.plot(kp2[:,0], kp2[:,1], 'r*')
ax2.set_title('(b)')
ax2.axis('off')
plt.show()

### SIFT descriptors ###
xy_pair1 = list(kp1)
cv2_kpt1= []
for i in range(len(xy_pair1)):
    cv2_kpt1.append(cv2.KeyPoint(xy_pair1[i][0], xy_pair1[i][1], 1))
#print("cv2_kpt1 ",cv2_kpt1)

xy_pair2 = list(kp2)
cv2_kpt2= []
for i in range(len(xy_pair2)):
    cv2_kpt2.append(cv2.KeyPoint(xy_pair2[i][0], xy_pair2[i][1], 1))
#print("cv2_kpt2 ",cv2_kpt2)

if method == 'sift' :
    _, des1 = sift.compute(img1_gray, cv2_kpt1)
    _, des2 = sift.compute(img2_gray, cv2_kpt2)

### normalise ###
for i in range(des1.shape[0]):
    des1[i, :] = des1[i, :] / np.linalg.norm(des1[i,:], ord=2)
for i in range(des2.shape[0]):
    des2[i, :] = des2[i, :] / np.linalg.norm(des2[i,:], ord=2)

### euclidean distance ###
dist = scipy.spatial.distance_matrix(des1, des2)

### matching points ###
match_kp1 =[]
dis_matrix = dist.copy()
for i in range(dist.shape[0]):
    match = np.argmin(dis_matrix[i,:])
    if i == np.argmin(dis_matrix[:,match]) and dis_matrix[i,match] <= 0.9:
        match_kp1.append(tuple([i, match]))
match_kp= np.array(match_kp1)

### Visualize the results ###
fig, (ax) = plt.subplots( constrained_layout = False ,figsize=(10, 5))
if  img1_rgb.shape[0] >  img2_rgb.shape[0]:
    white_strip = (np.ones(( img2_rgb.shape[0], 100, 3))*255).astype(np.uint8)
    combined_img = np.hstack(( img1_rgb[: img2_rgb.shape[0]], white_strip,  img2_rgb))
elif  img1_rgb.shape[0] <  img2_rgb.shape[0]:
    white_strip = (np.ones(( img1_rgb.shape[0], 100, 3))*255).astype(np.uint8)
    combined_img = np.hstack(( img1_rgb, white_strip,  img2_rgb[: img1_rgb.shape[0]]))
else:
    white_strip = (np.ones(( img1_rgb.shape[0], 100, 3))*255).astype(np.uint8)
    combined_img = np.hstack(( img1_rgb, white_strip,  img2_rgb))
ax.imshow(combined_img)
color = 'red'
ax.plot( [ kp1[match_kp[:,0], 0], img1_rgb.shape[1] + white_strip.shape[1] + kp2[match_kp[:,1], 0]  ],
        [ kp1[match_kp[:,0], 1], kp2[match_kp[:,1], 1] ],
        color=color, marker='*', linestyle='-', linewidth=1, markersize=5)
ax.axis('off')
plt.show()


### RANSAC ###

candidate_model_list = []
used_samples = []
for i in range( 1000):
    # Sample random matching pairs
    indices_of_indices = np.random.choice(match_kp.shape[0], size= 3, replace=False)
    if tuple(indices_of_indices) in used_samples:
        continue
    used_samples.append(tuple(indices_of_indices))
    sampled_kpt_pair_indices = match_kp[indices_of_indices]
    sampled_kp1, sampled_kp2  =  kp1[sampled_kpt_pair_indices[:,0]],  kp2[sampled_kpt_pair_indices[:,1]]

    # Fit a linear model
    sampled_kp1 = np.append(sampled_kp1, np.ones(( 3, 1)), axis=1)
    sampled_kp2 = np.append(sampled_kp2, np.ones(( 3, 1)), axis=1)
    model = np.linalg.lstsq(sampled_kp2, sampled_kp1, rcond=None)[0]

    # Check how many other points lie within the tolerance zone
    inlier_count = 0
    inlier_indices = []
    for pair_indices in match_kp:
        if pair_indices not in sampled_kpt_pair_indices:
            pt2 =  kp2[pair_indices[1]]
            pt1_true = np.append( kp1[pair_indices[0]], [1])
            pt1_hypothesis = np.dot(np.append(pt2, [1]), model)

            # Calculate error(euc distance b/w prediction and truth, in image coordinates)
            dist_from_model = np.linalg.norm(pt1_true-pt1_hypothesis,ord=2)

            if dist_from_model <= 40:
                #print(dist_from_model)
                inlier_count += 1
                inlier_indices.append(pair_indices)

    # Calculate avg. inlier residual for this model
    if inlier_count == 0:
        continue
    inlier_indices = np.array(inlier_indices)
    inlier_kp1, inlier_kp2 =  kp1[inlier_indices[:,0]],  kp2[inlier_indices[:,1]]
    inlier_kp1 = np.append(inlier_kp1, np.ones((inlier_kp1.shape[0],1)), axis=1)
    inlier_kp2 = np.append(inlier_kp2, np.ones((inlier_kp2.shape[0],1)), axis=1)

    hypothesis = np.dot(inlier_kp2, model)
    inlier_residuals = np.sum(np.square(inlier_kp1-hypothesis), axis=1)
    avg_inlier_residual = np.mean(inlier_residuals, axis=0)

    # Apply threshold and Refit
    if inlier_count >=  15:
        # The model is good -- Fit the model on all the inliers
        candidate_model, _, _, _ = np.linalg.lstsq(inlier_kp2, inlier_kp1, rcond=None)
        candidate_model_list.append(tuple([candidate_model, avg_inlier_residual, inlier_indices]))

# Choose the best model (one with least residual sum value)
if len(candidate_model_list) == 0:
    raise Exception("Couldn't find a good model for the given configuration")

candidate_model_list = sorted(candidate_model_list, key=lambda c: c[1])
affine_matrix, avg_residual, inlier_indices, = candidate_model_list[0]

print("Number of inliers: ", inlier_indices.shape[0])
print("Number of outliers: ", match_kp.shape[0] - inlier_indices.shape[0])
print("Average residual: ", avg_residual)

### Visualize the results ###

fig, (ax) = plt.subplots( constrained_layout = False ,figsize=(10, 5))
if  img1_rgb.shape[0] >  img2_rgb.shape[0]:
    white_strip = (np.ones(( img2_rgb.shape[0], 100, 3))*255).astype(np.uint8)
    combined_img = np.hstack(( img1_rgb[: img2_rgb.shape[0]], white_strip,  img2_rgb))
elif  img1_rgb.shape[0] <  img2_rgb.shape[0]:
    white_strip = (np.ones(( img1_rgb.shape[0], 100, 3))*255).astype(np.uint8)
    combined_img = np.hstack(( img1_rgb, white_strip,  img2_rgb[: img1_rgb.shape[0]]))
else:
    white_strip = (np.ones(( img1_rgb.shape[0], 100, 3))*255).astype(np.uint8)
    combined_img = np.hstack(( img1_rgb, white_strip,  img2_rgb))
ax.imshow(combined_img)
inlier_color = 'red'
outlier_color = 'blue'
outlier_indices = []
for indices in match_kp:
    if indices not in inlier_indices:
        outlier_indices.append(indices)
outlier_indices = np.array(outlier_indices)
ax.plot( [ kp1[outlier_indices[:,0], 0], img1_rgb.shape[1] + white_strip.shape[1] + kp2[outlier_indices[:,1], 0]  ],
            [ kp1[outlier_indices[:,0], 1], kp2[outlier_indices[:,1], 1] ],
            color=outlier_color, marker='*', linestyle='-', linewidth=1, markersize=5)
ax.plot( [ kp1[inlier_indices[:,0], 0], img1_rgb.shape[1] + white_strip.shape[1] + kp2[inlier_indices[:,1], 0]  ],
            [ kp1[inlier_indices[:,0], 1], kp2[inlier_indices[:,1], 1] ],
            color=inlier_color, marker='*', linestyle='-', linewidth=1, markersize=5)
ax.axis('off')
plt.show()

### Warp Image ###

width = img1_rgb.shape[1] + img2_rgb.shape[1]
height = max(img1_rgb.shape[0], img2_rgb.shape[0])
result = cv2.warpPerspective(img2_rgb, affine_matrix.T, (width, height))
result[0:img1_rgb.shape[0], 0:img1_rgb.shape[1]] = img1_rgb


### Visualize the results ###

fig, (ax) = plt.subplots( constrained_layout = False ,figsize=(10, 5))
ax.imshow(result)
ax.set_title('Stitched Image')
ax.axis('off')
plt.show()








