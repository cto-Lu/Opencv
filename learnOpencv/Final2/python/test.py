import cv2 as cv
import numpy as np
import copy
img = cv.imread('./53.jpg')

height = img.shape[0]
width = img.shape[1]
black = np.zeros((height, width, 3), np.uint8)
cv.rectangle(black, (178,176),(270,261), (255, 255, 255), thickness=-1)
ROI = cv.bitwise_and(img, black)
cv.imshow('ROI', ROI)
gray2 = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)

cv.GaussianBlur(gray2, (5, 5), 2)
ret, thresh = cv.threshold(gray2, 168, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE )
cv.drawContours(img, contours, -1, (0,255,0), 1)
x,y,w,h = cv.boundingRect(contours[2])
rectangle = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
cv.imshow('img', img)
cv.imshow('thresh', thresh)
cv.waitKey(0)