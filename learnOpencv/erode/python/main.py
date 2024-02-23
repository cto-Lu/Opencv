import cv2 as cv
import numpy as np
image = cv.imread("../python/24.png")
kernal = np.ones((30,30),np.uint8)
dilate = cv.dilate(image,kernal)
erode = cv.erode(image,kernal)
open = cv.morphologyEx(image,cv.MORPH_OPEN,kernal)
close = cv.morphologyEx(image,cv.MORPH_CLOSE,kernal)
cv.imshow("image",image)
cv.imshow("dilate",dilate)
cv.imshow("erode",erode)
cv.imshow("open",open)
cv.imshow("close",close)
cv.waitKey(0)