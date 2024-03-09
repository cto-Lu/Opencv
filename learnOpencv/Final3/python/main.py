import cv2 as cv
import copy
import numpy as np

img = cv.imread('./54.jpg')
copy = copy.deepcopy(img)
transfer = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('hsv', transfer)
img2 = transfer
red1 = cv.inRange(transfer, np.array([0, 50, 50]), np.array([10, 255, 255]))
red2 = cv.inRange(transfer, np.array([160, 40, 40]), np.array([180,255, 255]))
red3 = cv.addWeighted(red1, 0.5, red2, 0.5, 0)
element = np.ones((15, 15), np.uint8)
open = cv.morphologyEx(red3,cv.MORPH_OPEN, element)
cv.imshow("red", open)
contours, hierarchy = cv.findContours(open, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )
#cv.drawContours(copy, contours, -1, (0,255,0), 1)
x,y,w,h = cv.boundingRect(contours[0])
rectangle = cv.rectangle(copy,(x,y),(x+w,y+h),(0,255,255),2)
cv.imshow("rectangle", rectangle)
cv.waitKey(0)

