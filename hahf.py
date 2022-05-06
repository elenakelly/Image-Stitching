
import cv2

cereal = cv2.imread('images/leftimage.png') #query image
cereals = cv2.imread('images/rightimage.png') #train image


cv2.imshow('cereal', cereal)
cv2.waitKey(0)
cv2.imshow('cereals', cereals)
cv2.waitKey(0)

bf = cv2.BFMatcher()
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(cereal, None)
kp2, des2 = sift.detectAndCompute(cereals, None)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img = cv2.drawMatches(cereal, kp1, cereals, kp2, matches[:100], None, flags=2)
cv2.imshow('SIFT', img)
cv2.waitKey(0)
print(matches)

surf = cv2.xfeatures2d.SURF_create()
kp1, des1 = surf.detectAndCompute(cereal, None)
kp2, des2 = surf.detectAndCompute(cereals, None)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img = cv2.drawMatches(cereal, kp1, cereals, kp2, matches[:100], None, flags=2)
cv2.imshow('SURF', img)
#cv2.waitKey(0)
