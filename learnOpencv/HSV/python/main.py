import cv2 as cv
import numpy as np
image = cv.imread("../python/19.png")
image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
cv.imshow("pic", image)
blue = cv.inRange(image, np.array([100, 50 ,50]), np.array([130, 255, 255]))
cv.imshow("blue",blue)
red = cv.inRange(image, np.array([0, 50, 50]), np.array([10, 255, 255]))
cv.imshow("red",red)
cv.waitKey(0)

