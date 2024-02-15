import cv2 as cv
import numpy as np
image = cv.imread("/home/stoair/Opencv/Images/1.png", 4)
cv.namedWindow("initial",cv.WINDOW_NORMAL)
cv.namedWindow("after",cv.WINDOW_NORMAL)
height = image.shape[0]
width = image.shape[1]
cv.imshow("initial",image)
for i in range(height):
	for j in range(width):
		(b, g, r) = image[i, j]
		ave = b/3 + g/3 + r/3
		image[i, j] = (ave, ave, ave)

cv.imshow("after",image) 
cv.waitKey(0)
