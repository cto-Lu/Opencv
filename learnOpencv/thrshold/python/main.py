import cv2 as cv
import numpy as np
image = cv.imread("/home/stoair/Opencv/Images/1.png", 4)
image2 = cv.imread("/home/stoair/Opencv/Images/1.png", 4)
cv.namedWindow("initial",cv.WINDOW_NORMAL)
cv.namedWindow("after",cv.WINDOW_NORMAL)
cv.namedWindow("another",cv.WINDOW_NORMAL)

x=200
y=100

height = image.shape[0]
width = image.shape[1]
cv.imshow("initial",image)
for i in range(height):
	for j in range(width):
		(b, g, r) = image[i, j]
		ave = b/3 + g/3 + r/3
		if ave > x:
			ave = 255
		else:
			ave = 0
		image[i, j] = (ave, ave, ave)
for i in range(height):
	for j in range(width):
		(b, g, r) = image2[i, j]
		ave = b/3 + g/3 + r/3
		if ave > y:
			ave = 255
		else:
			ave = 0
		image2[i, j] = (ave, ave, ave)

cv.imshow("after",image) 
cv.imshow("another",image2) 
cv.waitKey(0)
