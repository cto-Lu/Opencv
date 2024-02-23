import cv2 as cv
import numpy as np
import copy
image = cv.imread("/home/stoair/Opencv/Images/1.png", 4)
image2 = cv.imread("/home/stoair/Opencv/Images/1.png", 4)

x=200
y=100

height = image.shape[0]
width = image.shape[1]

img = image
for i in range(height):
	for j in range(width):
		(b, g, r) = img[i, j]
		ave = b/3 + g/3 + r/3
		if ave > x:
			ave = 255
		else:
			ave = 0
		img[i, j] = (ave, ave, ave)
cv.imshow("shallow", image)
img2 = copy.deepcopy(image2)
for i in range(height):
	for j in range(width):
		(b, g, r) = img2[i, j]
		ave = b/3 + g/3 + r/3
		if ave > y:
			ave = 255
		else:
			ave = 0
		img2[i, j] = (ave, ave, ave)

cv.imshow("deep",image2)
cv.waitKey(0)
